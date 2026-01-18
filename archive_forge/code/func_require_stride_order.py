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
@classmethod
def require_stride_order(cls, x, order):
    if x.get_numel() == 0:
        return x
    if is_storage_and_layout(x):
        while isinstance(x.get_layout(), AliasedLayout):
            x = x.get_layout().view
        if isinstance(x.get_layout(), FlexibleLayout):
            as_storage_and_layout(x, freeze=True, want_contiguous=False, stride_order=order)
            return x
        elif isinstance(x.get_layout(), FixedLayout) and x.get_layout().is_stride_ordered(order):
            return x
        elif isinstance(x.get_layout(), MutationLayout):
            if isinstance(x.get_layout().real_layout(), FlexibleLayout):
                raise AssertionError("the MutationLayout's real layout shouldn't be FlexibleLayout")
            elif isinstance(x.get_layout().real_layout(), FixedLayout) and x.get_layout().real_layout().is_stride_ordered(order):
                return x
    if isinstance(x, InputBuffer) and x.get_layout().is_stride_ordered(order):
        return x
    if isinstance(x, TensorBox) and isinstance(x.data, BaseView) and (not isinstance(x.data, ReinterpretView)) and is_storage_and_layout(x.unwrap_view()) and (not isinstance(x.unwrap_view().data, ExternKernelAlloc)):
        try:
            x.data = cls.convert_to_reinterpret_view(x.data)
            return cls.require_stride_order(x, order)
        except NotImplementedError:
            pass
    x = cls.copy_input(x)
    as_storage_and_layout(x, freeze=True, want_contiguous=False, stride_order=order)
    assert is_stride_order_storage_and_layout(x, order)
    return x