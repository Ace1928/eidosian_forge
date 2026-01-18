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
def build_With(ctx, stmt):
    r = ctx.make_range(stmt.lineno, stmt.col_offset, stmt.col_offset + len('with'))
    if is_torch_jit_ignore_context_manager(stmt):
        if not _IS_ASTUNPARSE_INSTALLED:
            raise RuntimeError('torch.jit._IgnoreContextManager requires installing Python library `astunparse`,                                   please install it in your Python environment')
        assign_ast = build_ignore_context_manager(ctx, stmt)
        return build_stmt(ctx, assign_ast)
    return With(r, build_withitems(ctx, stmt.items), build_stmts(ctx, stmt.body))