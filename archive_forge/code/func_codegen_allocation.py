import collections
import contextlib
import dataclasses
import functools
import inspect
import os
import re
from itertools import chain, count
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import sympy
from sympy import Expr
import torch
from torch._dynamo.utils import counters, dynamo_timed
from torch._inductor.codecache import get_cpp_wrapper_cubin_path_name
from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols, SymTypes
from torch.fx.node import _get_qualified_name
from torch.utils._sympy.singleton_int import SingletonInt
from .. import codecache, config, ir
from ..codecache import CudaKernelParamCache
from ..ir import ComputedBuffer, InputBuffer, ReinterpretView
from ..triton_heuristics import grid as default_grid
from ..utils import (
from ..virtualized import V
from .common import CodeGen, DeferredLine, IndentedBuffer, PythonPrinter
from .triton_utils import config_of, signature_to_meta
def codegen_allocation(self, buffer):
    assert buffer.get_workspace_size() == 0, 'Only support zero workspace size for now!'
    name = buffer.get_name()
    if name in V.graph.removed_buffers or name in self.allocated:
        return
    self.allocated.add(name)
    if isinstance(buffer, (ir.ExternKernelAlloc, ir.MultiOutput)):
        return
    layout = buffer.get_layout()
    if isinstance(layout, ir.MutationLayout):
        return
    if isinstance(layout, ir.AliasedLayout):
        assert isinstance(layout.view, ir.ReinterpretView), f'unexpected {type(layout.view)}: {layout.view}'
        self.codegen_allocation(layout.view.data)
        self.codegen_deferred_allocation(name, layout)
        return
    self.writeline(AllocateLine(self, buffer))