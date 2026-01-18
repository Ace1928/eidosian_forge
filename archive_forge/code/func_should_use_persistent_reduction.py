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
def should_use_persistent_reduction(self) -> bool:
    """
        Heuristic to set self.persistent_reduction and add guards
        if needed.
        """
    if not (self.inside_reduction and config.triton.persistent_reductions):
        return False
    threshold = {ReductionHint.INNER: 1024}.get(self.reduction_hint, 64)
    last_numel = self.numels[-1]
    if not isinstance(last_numel, (int, sympy.Integer)):
        return False
    hint = V.graph.sizevars.size_hint(last_numel)
    if hint > threshold:
        return False
    V.graph.sizevars.guard_leq(self.numels[-1], next_power_of_2(hint))
    return True