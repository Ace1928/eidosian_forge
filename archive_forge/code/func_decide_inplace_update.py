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
def decide_inplace_update(self):
    """
        Decide if there should be inplace updates for the node
        and record the decision in the active kernel.
        """
    if not self.node.should_allocate():
        return
    if isinstance(self, (SchedulerNode,)) and (self.node.get_alias_names() or self.node.get_mutation_names()):
        return
    if (isinstance(self, (SchedulerNode,)) or (isinstance(self, ExternKernelSchedulerNode) and isinstance(self.node, (ir.AllReduce, ir.InPlaceHint)))) and config.inplace_buffers and (not isinstance(V.kernel, torch._inductor.codegen.triton.TritonKernel) or getattr(V.kernel, 'mutations', None) is not None):
        from .codegen.wrapper import buffer_reuse_key
        ordered_reads = sorted(self.read_writes.reads, key=lambda x: x.name)
        for read in ordered_reads:
            input_node: Optional[BaseSchedulerNode] = self.scheduler.name_to_node.get(read.name)
            if input_node and V.graph.wrapper_code.can_reuse(input_node, self):
                assert input_node.users is not None
                remaining_uses = [x for x in input_node.users if x.node.get_name() not in self.scheduler.available_buffer_names]
                if len(remaining_uses) == 1 and remaining_uses[0].can_inplace and (remaining_uses[0].node is self) and (not isinstance(input_node.node.get_layout(), (ir.MultiOutputLayout, ir.MutationLayout, ir.AliasedLayout))) and (not (isinstance(input_node.node, ir.FallbackKernel) and len(input_node.node.get_alias_names()) > 0)) and (buffer_reuse_key(input_node.node) == buffer_reuse_key(self.node)):
                    if hasattr(V.kernel, 'args'):
                        V.kernel.args.make_inplace(input_node.get_name(), self.get_name())
                        if isinstance(V.kernel, torch._inductor.codegen.triton.TritonKernel):
                            V.kernel.mutations.add(input_node.get_name())
                            V.kernel.mutations.add(self.get_name())
                        self.last_usage.discard(input_node.get_name())
                        V.kernel.inplace_update_buffers[self.get_name()] = input_node.get_name()
                    break