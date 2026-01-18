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
def get_fill_order(self):
    """
        If our layout is still flexible, try to determine the stride order based on stride orders of reads.

        TODO(jansel): A better algorithm here would look at downstream consumers of this
                      value and try to do global graph-level layout optimization.
                      This is also something just begging to be autotuned.
        """
    if isinstance(self.layout, FlexibleLayout):
        (index_vars, reduction_vars), _ = dependencies.index_vars_squeeze(self.data.get_size(), self.data.get_reduction_size())
        reads = self.get_read_writes().reads
        reads_bufs = [V.graph.name_to_buffer[r.name] if r.name in V.graph.name_to_buffer.keys() else None for r in reads]
        assert all((isinstance(r, (dependencies.StarDep, dependencies.MemoryDep)) for r in reads))
        reads = [sympy_subs(r.index, {v: sympy.Integer(0) for v in reduction_vars if v != 0}) for r in reads if isinstance(r, dependencies.MemoryDep)]
        if reads:
            stride_lengths = [V.graph.sizevars.stride_hints(expr, index_vars) for expr in reads]
            from .scheduler import pick_loop_order
            return pick_loop_order(stride_lengths, self.get_size())
    return None