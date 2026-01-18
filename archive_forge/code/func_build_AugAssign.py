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
def build_AugAssign(ctx, stmt):
    lhs = build_expr(ctx, stmt.target)
    rhs = build_expr(ctx, stmt.value)
    op = type(stmt.op)
    if op in StmtBuilder.augassign_map:
        op_token = StmtBuilder.augassign_map[op]
    else:
        raise NotSupportedError(find_before(ctx, rhs.range().start, '=', offsets=(-1, 0)), 'unsupported kind of augmented assignment: ' + op.__name__)
    return AugAssign(lhs, op_token, rhs)