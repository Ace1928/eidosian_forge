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
def build_Constant(ctx, expr):
    value = expr.value
    if value is None or isinstance(value, bool):
        return ExprBuilder.build_NameConstant(ctx, expr)
    if isinstance(value, (int, float, complex)):
        return ExprBuilder.build_Num(ctx, expr)
    elif isinstance(value, str):
        return ExprBuilder.build_Str(ctx, expr)
    elif isinstance(value, type(Ellipsis)):
        return ExprBuilder.build_Ellipsis(ctx, expr)
    else:
        error_range = ctx.make_range(expr.lineno, expr.col_offset, expr.col_offset + len(str(value)))
        raise FrontendError(error_range, 'Unknown Constant expression type')