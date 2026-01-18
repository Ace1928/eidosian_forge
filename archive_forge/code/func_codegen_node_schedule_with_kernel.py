from __future__ import annotations
import collections
import contextlib
import dataclasses
import functools
import itertools
import logging
import math
import operator
import os
import textwrap
from typing import Any, Counter, Dict, Iterable, List, Optional, Set, Tuple, Union
import sympy
import torch
import torch._logging
from torch._prims_common import is_integer_dtype
from torch.utils._sympy.functions import FloorDiv, ModularIndexing
from torch.utils._sympy.value_ranges import ValueRanges
from ..._dynamo.utils import counters
from .. import config, ir, scheduler
from ..codecache import code_hash, get_path, PyCodeCache
from ..dependencies import MemoryDep, StarDep
from ..ir import IRNode, ReductionHint, TritonTemplateBuffer
from ..optimize_indexing import indexing_dtype_strength_reduction
from ..scheduler import BaseScheduling, WhyNoFuse
from ..triton_heuristics import AutotuneHint
from ..utils import (
from ..virtualized import ops, V
from ..wrapper_benchmark import get_kernel_category_by_source_code
from .common import (
from .triton_utils import config_of, signature_of, signature_to_meta
def codegen_node_schedule_with_kernel(self, node_schedule, kernel):

    def current_reduction_nodes(nodes):
        return itertools.takewhile(lambda n: n is not DisableReduction, nodes)
    with kernel:
        stack = contextlib.ExitStack()
        kernel.set_last_usage(current_reduction_nodes(node_schedule))
        for node in node_schedule:
            if node not in (EnableReduction, DisableReduction):
                node.decide_inplace_update()
        for i, node in enumerate(node_schedule):
            if node is DisableReduction:
                stack.enter_context(kernel.disable_reduction())
            elif node is EnableReduction:
                stack.close()
                kernel.set_last_usage(current_reduction_nodes(node_schedule[i:]))
            else:
                indexing_dtype_strength_reduction(node._body)
                index_vars = kernel.split_and_set_ranges(node.get_ranges())
                node.codegen(index_vars)