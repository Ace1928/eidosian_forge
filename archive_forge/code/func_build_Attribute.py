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
def build_Attribute(ctx, expr):
    base = build_expr(ctx, expr.value)
    source = ctx.source.encode('utf-8')

    def get_char(index):
        return chr(source[index])
    start_pos = base.range().end + 1
    while get_char(start_pos) in string.whitespace:
        start_pos += 1
    end_pos = start_pos + len(expr.attr)
    name_range = ctx.make_raw_range(start_pos, end_pos)
    return Select(base, Ident(name_range, expr.attr))