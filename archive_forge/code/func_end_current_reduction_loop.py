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
@contextlib.contextmanager
def end_current_reduction_loop():
    if current_loop_writes:
        for other_node in nodes[index + 1:]:
            if node not in done and fits_in_main_body(other_node) and (not current_loop_writes & other_node.ancestors):
                done.add(node)
                current_loop_writes.add(node.get_name())
                is_current_reductions.add(node.is_reduction())
                node_schedule.append(node)
    if node_schedule and node_schedule[-1] is EnableReduction:
        node_schedule.pop()
    else:
        node_schedule.append(DisableReduction)
    yield
    node_schedule.append(EnableReduction)
    current_loop_writes.clear()
    is_current_reductions.clear()