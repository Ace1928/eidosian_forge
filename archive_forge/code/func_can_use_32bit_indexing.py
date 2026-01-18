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
@staticmethod
def can_use_32bit_indexing(numel: sympy.Expr, buffers: Iterable[Union[ir.Buffer, ir.TensorBox]]) -> bool:
    int_max = torch.iinfo(torch.int32).max
    size_hint = V.graph.sizevars.size_hint
    has_hint = V.graph.sizevars.shape_env.has_hint

    def within_32bit(e):
        if V.graph.sizevars.is_expr_static_and_true(e <= int_max):
            return True
        return has_hint(e) and size_hint(e) <= int_max
    if not within_32bit(numel):
        return False
    buf_sizes = [buf.get_layout().storage_size() for buf in buffers if not isinstance(buf.get_layout(), ir.MultiOutputLayout)]
    if not all((within_32bit(size) for size in buf_sizes)):
        return False
    V.graph.sizevars.guard_leq(numel, int_max)
    for size in buf_sizes:
        V.graph.sizevars.guard_leq(size, int_max)
    return True