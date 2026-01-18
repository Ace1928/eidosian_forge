import ast
import dataclasses
import inspect
import re
import string
import sys
from collections import namedtuple
from textwrap import dedent
from typing import List, Tuple  # noqa: F401
import torch
import torch.jit.annotations
from torch import _jit_internal
from torch._C._jit_tree_views import (
from torch._jit_internal import (  # noqa: F401
from torch._sources import (
from torch.jit._dataclass_impls import DATACLASS_MAGIC_METHODS
from torch.jit._monkeytype_config import get_qualified_name, monkeytype_trace
def build_ExtSlice(ctx, base, extslice):
    sub_exprs = []
    for expr in extslice.dims:
        sub_type = type(expr)
        if sub_type is ast.Index:
            sub_exprs.append(build_Index(ctx, base, expr))
        elif sub_type is ast.Slice:
            sub_exprs.append(build_SliceExpr(ctx, base, expr))
        elif sub_type is ast.Ellipsis:
            sub_exprs.append(Dots(base.range()))
        else:
            raise NotSupportedError(base.range(), f'slicing multiple dimensions with {sub_type} not supported')
    return sub_exprs