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
@staticmethod
def build_Call(ctx, expr):
    func = build_expr(ctx, expr.func)
    args = [build_expr(ctx, py_arg) for py_arg in expr.args]
    if hasattr(expr, 'starargs') and expr.starargs:
        stararg_expr = build_expr(ctx, expr.starargs)
        args += [Starred(stararg_expr.range(), stararg_expr)]
    kwargs = []
    for kw in expr.keywords:
        kw_expr = build_expr(ctx, kw.value)
        if not kw.arg:
            raise NotSupportedError(kw_expr.range(), 'keyword-arg expansion is not supported')
        kwargs.append(Attribute(Ident(kw_expr.range(), kw.arg), kw_expr))
    return Apply(func, args, kwargs)