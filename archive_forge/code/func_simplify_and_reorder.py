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
def simplify_and_reorder(x_vars, support_vars, sizes, reordering_reindex=None):
    sizes, reindex0, reindex1 = self._apply_loop_reordering(x_vars, support_vars, sizes, memory_addrs, reordering_reindex)
    x_vars = reindex0(x_vars)
    sizes, reindex2, prune = V.graph.sizevars._simplify_loops(x_vars, sizes, index_prevent_reordering(index_formulas, x_vars, sizes))
    x_vars = prune(x_vars)
    reindex = fuse_reindexing(reindex1, reindex2)
    return (sizes, reindex, reindex1)