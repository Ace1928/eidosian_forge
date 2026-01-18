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
def free_buffers(self):
    """Free any buffers that are no longer needed"""
    for name in sorted(self.buffer_names_to_free - V.graph.removed_buffers - V.graph.wrapper_code.freed):
        if name in self.name_to_node:
            node = self.name_to_node[name]
            if node.can_free():
                V.graph.wrapper_code.codegen_free(node.node)
        elif name in V.graph.graph_inputs:
            storage = V.graph.graph_inputs[name].data
            assert isinstance(storage, ir.StorageBox) and storage.is_input_buffer()
            V.graph.wrapper_code.codegen_free(storage.data)
    self.buffer_names_to_free.clear()