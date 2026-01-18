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
def is_stride_ordered(self, order):
    assert len(self.stride) == len(order)
    non_1_indices = [i for i, dim in enumerate(self.size) if V.graph.sizevars.size_hint(dim, fallback=2) != 1]
    stride = [self.stride[i] for i in non_1_indices]
    order = [order[i] for i in non_1_indices]

    def sorted_indices(arr):
        sorted_arr = sorted(arr)
        return [sorted_arr.index(element) for element in arr]
    order = sorted_indices(order)
    stride_ordered = [-1] * len(order)
    for i in range(len(order)):
        stride_ordered[order[i]] = V.graph.sizevars.size_hint(stride[i])
    for i in range(len(order) - 1):
        if stride_ordered[i] > stride_ordered[i + 1]:
            return False
    return True