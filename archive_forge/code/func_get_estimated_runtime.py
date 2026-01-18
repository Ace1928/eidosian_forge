import collections
import dataclasses
import functools
import itertools
import logging
import math
import os
import pprint
import textwrap
from typing import (
import sympy
import torch
from torch._dynamo.utils import dynamo_timed
from torch._inductor.metrics import get_metric_table, is_metric_table_enabled
from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols
from torch.utils._triton import has_triton
from . import comms, config, dependencies, ir, metrics
from .codegen.common import get_scheduling_for_device, Kernel
from .comm_analysis import estimate_nccl_collective_runtime
from .dependencies import StarDep, WeakDep
from .ir import ComputedBuffer, MultiOutput, MultiOutputLayout
from .sizevars import SimplifyIndexing
from .utils import (
from .virtualized import V
def get_estimated_runtime(self) -> float:
    """
        Returns estimated op runtime in nanoseconds (ns)
        """
    layout = None
    dtype = None
    if not hasattr(self, 'node') or not self.node:
        assert isinstance(self, (FusedSchedulerNode, ForeachKernelSchedulerNode)), f'type(self)={type(self)!r}'
        assert self.snodes
        if not self.snodes[0].node:
            return 0
        layout = self.snodes[0].node.get_layout()
        dtype = self.snodes[0].node.get_dtype()
    else:
        layout = self.node.get_layout()
        dtype = self.node.get_dtype()
    if 'cuda' != layout.device.type:
        return 0
    try:
        gpu_memory_bandwidth = get_gpu_dram_gbps()
        gpu_flops = get_device_tflops(dtype) * 10 ** 12
    except Exception:
        return 0
    if isinstance(self, ExternKernelSchedulerNode):
        assert isinstance(self.node, ir.ExternKernel), f'type(self.node)={type(self.node)!r}'
        op = kernel_name_to_op.get(getattr(self.node, 'kernel', ''), None)
        if op is not None:
            from torch._subclasses.fake_tensor import FakeTensorMode
            from torch.utils.flop_counter import FlopCounterMode
            with FakeTensorMode(), FlopCounterMode(display=False) as flop_counter_mode:
                from .ir import ir_node_to_tensor
                fake_inputs = [ir_node_to_tensor(input, guard_shape=False) for input in self.node.inputs]
                cls = self.node.__class__
                cls.process_kernel(op, *fake_inputs, **self.node.kwargs)
                factor = 1.0
                counted_flops = flop_counter_mode.get_total_flops()
                counted_bytes = self.get_read_write_buffers_sizes()
                compute_time = factor * counted_flops / gpu_flops * 1000000000.0
                transfer_time = counted_bytes / gpu_memory_bandwidth
                return max(compute_time, transfer_time)
    elif isinstance(self, FusedSchedulerNode) or isinstance(self.node, ComputedBuffer):
        return self.get_read_write_buffers_sizes() / gpu_memory_bandwidth
    if isinstance(self.node, ir.CollectiveKernel):
        return estimate_nccl_collective_runtime(self)
    elif isinstance(self.node, ir.Wait):
        return 0
    return 0