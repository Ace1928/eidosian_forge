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
@classmethod
def select_tiling(cls, node_schedule, numel, reduction_numel=sympy.Integer(1)):
    """
        Heuristics to decide how to tile kernels.
        Currently, we tile based on stride-1 dimensions.

        Returns:
            `(tile1, tile2, reduction_numel)` s.t. `tile1 * tile2 == numel`

        """
    if reduction_numel != 1 or config.triton.max_tiles <= 1:
        if perf_hint_log.level <= logging.WARNING:
            for node in EnableReduction.filter(node_schedule):
                if len(cls.candidate_tilings(node)) > 0:
                    perf_hint_log.info('reduction over non-contiguous dims')
                    break
        return (numel, reduction_numel)
    seen_names = set()
    candidate_tiles: Counter[Any] = collections.Counter()
    for node in EnableReduction.filter(node_schedule):
        for tiling in cls.candidate_tilings(node):
            if tiling.name in seen_names:
                continue
            seen_names.add(tiling.name)
            candidate_tiles[tiling.tiling] += tiling.score
    ranked_tilings = [tiling for tiling, score in candidate_tiles.most_common()]
    if config.triton.max_tiles >= 3:
        for i in range(1, len(ranked_tilings)):
            a0, a1 = ranked_tilings[0]
            b0, b1 = ranked_tilings[i]
            if V.graph.sizevars.size_hint(a1 - b1) == 0:
                continue
            if V.graph.sizevars.size_hint(a1 - b1) < 0:
                a0, a1 = ranked_tilings[i]
                b0, b1 = ranked_tilings[0]
            assert V.graph.sizevars.size_hint(a1 - b1) > 0
            if V.graph.sizevars.statically_known_multiple_of(a1, b1):
                tiling = (a0, FloorDiv(a1, b1), b1)
                ranked_tilings = [tiling] + ranked_tilings
                break
    if len(ranked_tilings) > 1:
        perf_hint_log.info('possibly bad tiling: %s', ranked_tilings)
    for tiled_groups in ranked_tilings:
        new_groups = (*tiled_groups, reduction_numel)
        if all((TritonKernel.is_compatible(new_groups, node.get_ranges()) for node in node_schedule if isinstance(node, scheduler.SchedulerNode))):
            return new_groups
    return (numel, reduction_numel)