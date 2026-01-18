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
def combine_contiguous_dims(self, index: sympy.Expr, tree: IterationRangesRoot):
    """
        More aggressive simplification to merge contiguous dims
        """
    if isinstance(index, (sympy.Integer, sympy.Symbol)):
        return index
    index_vars, sizes = tree.vars_and_sizes(index)
    if len(sizes) <= 1:
        return index
    new_sizes, reindex, prune = V.graph.sizevars._simplify_loops(index_vars, sizes, index_prevent_reordering([index], index_vars, sizes))
    if new_sizes == sizes:
        return index
    new_index_vars = tree.construct(new_sizes)
    new_index = sympy_subs(index, dict(zip(index_vars, reindex(new_index_vars))))
    return new_index