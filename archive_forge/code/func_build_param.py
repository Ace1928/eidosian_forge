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
def build_param(ctx, py_arg, self_name, kwarg_only, pdt_arg_type=None):
    name = py_arg.arg
    r = ctx.make_range(py_arg.lineno, py_arg.col_offset, py_arg.col_offset + len(name))
    if getattr(py_arg, 'annotation', None) is not None:
        annotation_expr = build_expr(ctx, py_arg.annotation)
    elif pdt_arg_type:
        annotation_expr = Var(Ident(r, pdt_arg_type))
    elif self_name is not None and name == 'self':
        annotation_expr = Var(Ident(r, self_name))
    else:
        annotation_expr = EmptyTypeAnnotation(r)
    return Param(annotation_expr, Ident(r, name), kwarg_only)