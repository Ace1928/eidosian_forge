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
def dense_size_str(self):
    sizes = []
    for tree in self.range_trees:
        if self.no_x_dim and tree.prefix == 'x':
            continue
        if tree.prefix != 'r' or self.inside_reduction:
            sizes.append(f'{tree.prefix.upper()}BLOCK')
        elif tree.prefix == 'r' and tree.numel != 1:
            sizes.append('1')
    if sizes[0:3] == ['ZBLOCK', 'YBLOCK', 'XBLOCK']:
        sizes[0:3] = reversed(sizes[0:3])
    if sizes[0:2] == ['YBLOCK', 'XBLOCK']:
        sizes[0:2] = reversed(sizes[0:2])
    return f'[{', '.join(sizes)}]'