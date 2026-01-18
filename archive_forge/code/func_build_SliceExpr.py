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
def build_SliceExpr(ctx, base, slice_expr):
    lower = build_expr(ctx, slice_expr.lower) if slice_expr.lower is not None else None
    upper = build_expr(ctx, slice_expr.upper) if slice_expr.upper is not None else None
    step = build_expr(ctx, slice_expr.step) if slice_expr.step is not None else None
    return SliceExpr(base.range(), lower, upper, step)