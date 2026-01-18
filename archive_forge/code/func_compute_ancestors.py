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
def compute_ancestors(self):
    """
        Populate each node.ancestors
        """
    name_to_ancestors: Dict[str, Set[str]] = {}
    for node in self.nodes:
        ancestors = set()
        for dep in node.unmet_dependencies:
            ancestors.add(dep.name)
            ancestors |= name_to_ancestors[dep.name]
        name_to_ancestors[node.get_name()] = ancestors
        node.ancestors = ancestors
    for order, node in enumerate(self.nodes):
        node.min_order = order
        node.max_order = order