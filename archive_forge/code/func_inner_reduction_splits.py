import collections
import contextlib
import dataclasses
import functools
import itertools
import logging
import re
import textwrap
import traceback
from contextlib import nullcontext
from enum import Enum
from functools import partial
from inspect import signature
from typing import (
from unittest.mock import patch
import sympy
from sympy import Expr, Integer
import torch._export.serde.schema as export_schema
import torch._logging
import torch.fx
import torch.utils._pytree as pytree
from torch._dynamo.device_interface import get_interface_for_device
from torch._dynamo.utils import identity
from torch._export.serde.serialize import GraphModuleSerializer
from torch._prims_common import (
from torch._subclasses.fake_tensor import get_schema_info
from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols, SymTypes
from torch.utils._sympy.functions import CleanDiv, FloorDiv, ModularIndexing
from . import config, dependencies
from .codegen.common import index_prevent_reordering
from .dependencies import (
from .utils import (
from .virtualized import ops, V
def inner_reduction_splits(reduction_numel_hint, numel_hint):
    num_warps = 8
    num_threads = 32 * num_warps
    if numel_hint >= 2 * num_sm:
        return 1
    if reduction_numel_hint <= 8192:
        return 1
    if reduction_numel_hint * numel_hint <= min_elements_per_device:
        split_size = min_elements_per_thread
    elif reduction_numel_hint * numel_hint < max_elements_per_device:
        target_blocks = num_sm * threads_per_sm // (2 * num_threads)
        blocks_per_output = (target_blocks + numel_hint - 1) // numel_hint
        tmp_split_size = (reduction_numel_hint + num_threads * blocks_per_output - 1) // (num_threads * blocks_per_output)
        divisors = sympy.divisors(reduction_numel_hint)
        closest = min(divisors, key=lambda x: abs(x - tmp_split_size))
        if abs(closest - tmp_split_size) < 30:
            split_size = max(closest, min_elements_per_thread)
        else:
            split_size = tmp_split_size
    else:
        divisors = sympy.divisors(reduction_numel_hint)
        closest = min(divisors, key=lambda x: abs(x - max_elements_per_thread))
        if abs(closest - max_elements_per_thread) < 50:
            split_size = closest
        else:
            split_size = max_elements_per_thread
    return (reduction_numel_hint + split_size * num_threads - 1) // (split_size * num_threads)