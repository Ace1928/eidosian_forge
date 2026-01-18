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
def split_and_set_ranges(self, lengths: List[List[sympy.Expr]]):
    """
        We may want to fuse `for i0 in s0*s1` into a tiled kernel with groups (s0, s1).

        To do this we need to split up the iteration space of i0 into something like:
            for i1 in s0:
              for i2 in s1:
                i0 = i1*s1 + i2
                ....

        This function matches and resplits lengths to the groups of
        this kernel to enable tiled + non-tiled fusions.
        """
    groups = [rt.numel for rt in self.range_trees]
    if not self.inside_reduction:
        groups[-1] = sympy.Integer(1)
    if len(lengths) == len(self.range_trees) and all((V.graph.sizevars.simplify(sympy_product(x) - g) == 0 for x, g in zip(lengths, groups))):
        return self.set_ranges(*lengths)
    new_ranges, return_getters_groups = self._split_iteration_ranges(groups, lengths)
    itervars = list(itertools.chain(*self.set_ranges(*new_ranges)))
    return [[fn(itervars) for fn in fns] for fns in return_getters_groups]