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
def score_fusion(self, node1: BaseSchedulerNode, node2: BaseSchedulerNode):
    """
        Assign a score (higher comes first) to the fusion of node1
        and node2.  When different fusions conflict with each other,
        this is the way we decide what order to run them in.

        Our current score is based on:
        - Estimate of the saved memory operations
        - Fusions closer together in original order
        """
    memory_score = self.score_fusion_memory(node1, node2)
    proximity_score = -max(abs(node1.min_order - node2.max_order), abs(node2.min_order - node1.max_order))
    return (node1.is_template() == config.epilogue_fusion_first and memory_score > 0, node1.is_reduction() == node2.is_reduction() and memory_score > 0, memory_score, proximity_score)