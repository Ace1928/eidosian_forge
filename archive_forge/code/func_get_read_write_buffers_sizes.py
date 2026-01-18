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
def get_read_write_buffers_sizes(self) -> int:
    """
        Counting the number of bytes accessed for a kernel is
        surprisingly tricky. In particular, there is a differentiation
        between 'theoretical' memory accesses and practical memory
        accesses. For example, a layernorm kernel may actually access an
        input 3 times, but in theory, it only needs to access its input
        once (and may be optimized to do so through say, persistent
        reductions)

        Another example is that even though a buffer is passed in, we may
        not access the entire buffer. This may occur if we are accessing
        a slice of the buffer. Another tricky case is for indirect
        indexing, where the amount of bytes accessed depends on the
        values of the input.

        What this function aims to compute is the memory accesses for
        worst-case inputs, best-case optimization. What this means is
        that for each buffer we compute the amount of potential accesses in two ways and take the minimum.

        1. Numel in ranges multiplied by number of deps the buffer has
        2. The buffer size
        """
    if isinstance(self, NopKernelSchedulerNode):
        return 0
    if isinstance(self, ExternKernelSchedulerNode) and isinstance(self.node, MultiOutput):
        return 0
    if isinstance(self, SchedulerNode):
        node_numel = V.graph.sizevars.size_hint(sympy_product(self.get_ranges()[0]) * sympy_product(self.get_ranges()[1]))
    else:
        node_numel = int(1000000000.0)
    buf_accesses = collections.defaultdict(list)
    for dep in self.read_writes.reads | self.read_writes.writes:
        buf_accesses[dep.name].append(dep)
    reads = {dep.name for dep in self.read_writes.reads}
    writes = {dep.name for dep in self.read_writes.writes}

    def is_materialized(buf, snodes):
        users = self.scheduler.name_to_node[buf].users
        buf_uses = {user.node for user in users}
        return len(buf_uses - set(snodes)) > 0
    if isinstance(self, FusedSchedulerNode):
        removed_buffers = {dep for dep in writes if not is_materialized(dep, self.snodes)}
        writes = writes - removed_buffers
        reads = reads - removed_buffers
    node_bytes = 0
    for buf_name in reads | writes:
        buf_accessed_elems = sum([node_numel for dep in buf_accesses[buf_name]])
        buf: Union[ir.Buffer, ir.TensorBox]
        if buf_name in V.graph.name_to_buffer:
            buf = V.graph.name_to_buffer[buf_name]
        elif buf_name in V.graph.graph_inputs:
            buf = V.graph.graph_inputs[buf_name]
        else:
            continue

        def get_buf_elems(buf):
            return V.graph.sizevars.size_hint(sympy_product(buf.get_size()))
        if isinstance(buf.layout, MultiOutputLayout):
            users = self.scheduler.name_to_node[buf.get_name()].users
            buf_elems = sum((get_buf_elems(user.node.node) for user in users))
        else:
            buf_elems = get_buf_elems(buf)
        node_bytes += min(buf_elems, buf_accessed_elems) * get_dtype_size(buf.get_dtype())
    return node_bytes