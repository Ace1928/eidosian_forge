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
def make_buffer_reuse(self, old, new, delete_old: bool):
    assert old.get_dtype() == new.get_dtype()
    old_name = old.get_name()
    new_name = new.get_name()
    del_line = ';'
    if old_name not in V.graph.get_output_names() and delete_old:
        del_line = f'; {self.make_buffer_free(old)}'
    if old.get_size() == new.get_size() and old.get_stride() == new.get_stride():
        if old_name in self.cached_thread_locals:
            self.cached_thread_locals.add(new_name)
        return self.codegen_exact_buffer_reuse(old_name, new_name, del_line)
    reinterpret_view = self.codegen_reinterpret_view(old, new.get_size(), new.get_stride(), 0, self.wrapper_call)
    if reinterpret_view in self.cached_thread_locals:
        self.cached_thread_locals.add(new_name)
    return f'{self.declare}{new_name} = {reinterpret_view}{del_line}  {self.comment} reuse'