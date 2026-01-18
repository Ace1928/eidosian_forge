import collections
import contextlib
import dataclasses
import functools
import itertools
import logging
import re
import textwrap
import traceback
from contextlib import nullcontext
from enum import Enum
from functools import partial
from inspect import signature
from typing import (
from unittest.mock import patch
import sympy
from sympy import Expr, Integer
import torch._export.serde.schema as export_schema
import torch._logging
import torch.fx
import torch.utils._pytree as pytree
from torch._dynamo.device_interface import get_interface_for_device
from torch._dynamo.utils import identity
from torch._export.serde.serialize import GraphModuleSerializer
from torch._prims_common import (
from torch._subclasses.fake_tensor import get_schema_info
from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols, SymTypes
from torch.utils._sympy.functions import CleanDiv, FloorDiv, ModularIndexing
from . import config, dependencies
from .codegen.common import index_prevent_reordering
from .dependencies import (
from .utils import (
from .virtualized import ops, V
def add_index_expr(self, expr: sympy.Expr, category, buf_name):
    getattr(self, category).append(expr)
    if buf_name is not None:
        getattr(self, f'{category}_name2expr')[buf_name] = expr
    if expr not in self.indexing_exprs_name:
        name = f'index{len(self.indexing_exprs)}'
        self.indexing_exprs_name[expr] = name
        self.indexing_exprs[name] = expr
    return self.indexing_exprs_name[expr]