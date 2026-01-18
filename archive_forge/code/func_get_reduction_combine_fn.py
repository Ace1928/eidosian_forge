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
def get_reduction_combine_fn(reduction_type, dtype):
    if reduction_type in REDUCTION_COMBINE_FN:
        combine_fn = REDUCTION_COMBINE_FN[reduction_type]
    elif reduction_type in {'argmax', 'argmin'}:

        def combine_fn(a, b):
            a_value, a_index = a
            b_value, b_index = b
            if reduction_type == 'argmin':
                mask = ops.lt(a_value, b_value)
            else:
                mask = ops.gt(a_value, b_value)
            equal = ops.eq(a_value, b_value)
            if is_float_dtype(dtype):
                a_isnan = ops.ne(a_value, a_value)
                b_isnan = ops.ne(b_value, b_value)
                mask = ops.logical_or(mask, ops.gt(a_isnan, b_isnan))
                equal = ops.logical_or(equal, ops.logical_and(a_isnan, b_isnan))
            mask = ops.logical_or(mask, ops.logical_and(equal, ops.lt(a_index, b_index)))
            return (ops.where(mask, a_value, b_value), ops.where(mask, a_index, b_index))
    elif reduction_type == 'welford_combine':

        def combine_fn(a, b):
            a_mean, a_m2, a_weight = a
            b_mean, b_m2, b_weight = b
            delta = b_mean - a_mean
            new_weight = a_weight + b_weight
            w2_over_w = b_weight / new_weight
            return (a_mean + delta * w2_over_w, a_m2 + b_m2 + delta * delta * a_weight * w2_over_w, new_weight)
    else:
        raise NotImplementedError(f'unknown reduction_type={reduction_type}')
    return combine_fn