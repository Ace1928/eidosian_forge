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
def build_JoinedStr(ctx, expr):
    s = ''
    args = []
    for value in expr.values:
        r = ctx.make_range(value.lineno, value.col_offset, value.col_offset + 1)
        if isinstance(value, ast.FormattedValue):
            if value.conversion != -1:
                raise NotSupportedError(r, "Don't support conversion in JoinedStr")
            if value.format_spec is not None:
                raise NotSupportedError(r, "Don't support formatting in JoinedStr")
            s += '{}'
            args.append(build_expr(ctx, value.value))
        elif isinstance(value, ast.Str):
            s += value.s
        else:
            raise NotSupportedError(r, 'Unsupported value in JoinedStr')
    r = ctx.make_range(expr.lineno, expr.col_offset, expr.col_offset + 1)
    return Apply(Select(StringLiteral(r, s), Ident(r, 'format')), args, [])