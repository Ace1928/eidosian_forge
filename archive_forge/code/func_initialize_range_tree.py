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
def initialize_range_tree(self, pid_cache):
    names = list(reversed(['xindex', 'yindex', 'zindex'][:len(self.numels) - 1])) + ['rindex']
    for i in range(len(self.numels)):
        pid_idx = i if names[i][0] == 'r' else 'xyz'.find(names[i][0])
        self.range_trees.append(IterationRangesRoot(names[i], self.numels[i], names[i][0], pid_idx, self, pid_cache))
    for tree in self.range_trees:
        if not tree.is_loop():
            tree.codegen_header(self.body, self.no_x_dim)
    if self.inside_reduction and self.range_trees[-1].is_loop():
        self.body.writeline(f'rbase = {self.range_trees[-1].ranges_code()}')