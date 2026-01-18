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
@functools.lru_cache(32)
def candidate_tilings(node):
    ranges, reduction_ranges = node.get_ranges()
    if len(ranges) <= 1:
        return ()
    rw = node.pointwise_read_writes()
    assert len(rw.range_vars) == len(ranges)
    dep_sources = [rw.reads, rw.writes]
    assert all((isinstance(dep, (MemoryDep, StarDep)) for dep in itertools.chain(*dep_sources)))
    deps = [dep for dep in itertools.chain(*dep_sources) if dep.name not in V.graph.removed_buffers and isinstance(dep, MemoryDep)]
    write_names = {dep.name for dep in rw.writes}
    tilings: List[CandidateTiling] = []
    for dep in deps:
        strides = V.graph.sizevars.stride_hints(dep.index, rw.range_vars)
        assert len(strides) == len(ranges)
        try:
            split = strides.index(1) + 1
            if split == len(ranges):
                continue
            if all((s == 0 for s in strides[split:])):
                continue
        except ValueError:
            continue
        tiled_groups = (V.graph.sizevars.simplify(sympy_product(ranges[:split])), V.graph.sizevars.simplify(sympy_product(ranges[split:])))
        score = V.graph.sizevars.size_hint(sympy_product((size for size, stride in zip(ranges, strides) if stride != 0)))
        if dep.name in write_names:
            score *= 2
        if CandidateTiling.is_good_size(tiled_groups[0]):
            score *= 2
        if CandidateTiling.is_good_size(tiled_groups[1]):
            score *= 2
        if V.graph.sizevars.size_hint(score - sympy_product(itertools.chain(ranges, reduction_ranges))) >= 0:
            tilings.append(CandidateTiling(tiled_groups, score, dep.name))
    return tilings