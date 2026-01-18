import hashlib
import logging
import operator
import os
import re
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Callable, DefaultDict, Dict, List, Optional, Set, Tuple
import sympy
import torch
import torch._logging
import torch.fx
from torch._decomp import get_decompositions
from torch._dynamo.utils import defake, dynamo_timed
from torch._logging import LazyString
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx.experimental.sym_node import magic_methods, method_to_operator
from torch.fx.experimental.symbolic_shapes import has_free_symbols, ShapeEnv, SymTypes
from torch.utils._mode_utils import no_dispatch
from . import config, ir
from .codegen.common import (
from .codegen.wrapper import CppWrapperCodeGen, CudaWrapperCodeGen, WrapperCodeGen
from .exc import (
from .ir import (
from .lowering import (
from .sizevars import SizeVarAllocator
from .utils import convert_shape_to_inductor, gather_origins, get_sympy_Expr_dtype
from .virtualized import V
def get_numel(self, buffer_name: str):
    from .ir import MultiOutputLayout
    if buffer_name in self.constants:
        return self.constants[buffer_name].numel()
    if buffer_name in self.name_to_buffer:
        buf = self.name_to_buffer[buffer_name]
        if isinstance(getattr(buf, 'layout', None), MultiOutputLayout):
            return 1
        return buf.get_numel()
    if buffer_name in self.graph_inputs:
        return self.graph_inputs[buffer_name].get_numel()
    raise KeyError(f'could not find {buffer_name}')