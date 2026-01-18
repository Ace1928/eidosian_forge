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
def codegen_extern_call(self, scheduler_node: ExternKernelSchedulerNode):
    assert isinstance(scheduler_node, ExternKernelSchedulerNode)
    with V.set_kernel_handler(Kernel(increase_kernel_count=False)):
        scheduler_node.decide_inplace_update()
        scheduler_node.allocate()
    node = scheduler_node.node
    assert isinstance(node, ir.ExternKernel), f'type(node)={type(node)!r}'
    node.codegen(V.graph.wrapper_code)
    self.free_buffers()