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
@staticmethod
def default_accumulator(reduction_type, dtype):
    if reduction_type in {'max', 'argmax'}:
        if is_float_dtype(dtype):
            return float('-inf')
        elif is_boolean_dtype(dtype):
            return 0
        else:
            return torch.iinfo(dtype).min
    if reduction_type in {'min', 'argmin'}:
        if is_float_dtype(dtype):
            return float('inf')
        elif is_boolean_dtype(dtype):
            return 1
        else:
            return torch.iinfo(dtype).max
    return {'sum': 0, 'prod': 1, 'xor_sum': 0, 'any': 0, 'welford_reduce': (0, 0, 0), 'welford_combine': (0, 0, 0)}[reduction_type]