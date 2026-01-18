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
def build_ListComp(ctx, stmt):
    r = ctx.make_range(stmt.lineno, stmt.col_offset, stmt.col_offset)
    if len(stmt.generators) != 1:
        raise NotSupportedError(r, 'Only a single generator is currently supported')
    if len(stmt.generators[0].ifs) != 0:
        raise NotSupportedError(r, 'Comprehension ifs are not supported yet')
    elt_expr = build_expr(ctx, stmt.elt)
    target_expr = build_expr(ctx, stmt.generators[0].target)
    iter_expr = build_expr(ctx, stmt.generators[0].iter)
    return ListComp(r, elt_expr, target_expr, iter_expr)