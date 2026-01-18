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
def build_NameConstant(ctx, expr):
    r = ctx.make_range(expr.lineno, expr.col_offset, expr.col_offset + len(str(expr.value)))
    if expr.value is True:
        return TrueLiteral(r)
    elif expr.value is False:
        return FalseLiteral(r)
    elif expr.value is None:
        return NoneLiteral(r)
    elif expr.value == Ellipsis:
        return Dots(r)
    else:
        raise ValueError('Name constant value unsupported: ' + str(expr.value))