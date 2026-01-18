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
def get_class_assigns(ctx, cls_ast):
    assigns = []

    def maybe_build_assign(builder, entry):
        nonlocal assigns
        try:
            assigns.append(builder(ctx, entry))
        except NotSupportedError:
            pass
    for entry in cls_ast.body:
        if isinstance(entry, ast.Assign):
            maybe_build_assign(StmtBuilder.build_Assign, entry)
        elif isinstance(entry, ast.AnnAssign):
            maybe_build_assign(StmtBuilder.build_AnnAssign, entry)
    return assigns