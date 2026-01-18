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
def is_broadcasted(self, index: sympy.Expr):
    if self.is_indirect_indexing(index):
        return False
    index_numels = [1] * len(self.numels)
    for symbol in index.free_symbols:
        if symbol not in self.range_tree_nodes:
            continue
        entry = self.range_tree_nodes[symbol]
        assert isinstance(entry.parent, IterationRangesRoot)
        index_numels[entry.parent.index] *= entry.length
    simplify = V.graph.sizevars.simplify
    return any((simplify(idx_range) != simplify(iter_range) for idx_range, iter_range in zip(index_numels, self.numels)))