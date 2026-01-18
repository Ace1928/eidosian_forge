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
def convert_to_reinterpret_view(cls, x):
    """
        In order to pass this to an extern kernel we need a
        ReinterpretView not a View.  This allows us to avoid some
        unneeded copies.
        """
    assert isinstance(x, BaseView)
    if isinstance(x, ReinterpretView):
        return x
    x.unwrap_view().freeze_layout()
    index_args, var_ranges = dependencies.index_vars_squeeze(x.get_size(), prefix='r')
    range_vars = index_args[0]
    index = x.make_indexer()(range_vars)
    index = V.graph.sizevars.simplify_with_ranges(index, var_ranges)
    strides = V.graph.sizevars.stride_vars(index, range_vars)
    offset = V.graph.sizevars.offset_var(index, range_vars)
    expected = sympy_dot(range_vars, strides) + offset
    if index != expected:
        log.debug('convert_to_reinterpret_view failed: stride=%s offset=%s index=%s', strides, offset, index)
        raise NotImplementedError()
    return ReinterpretView(data=x.data, layout=FixedLayout(device=x.get_device(), dtype=x.get_dtype(), size=x.get_size(), stride=strides, offset=offset))