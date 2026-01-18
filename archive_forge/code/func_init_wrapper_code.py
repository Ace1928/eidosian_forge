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
def init_wrapper_code(self):
    self.cuda = 'cuda' in self.device_types
    if self.cpp_wrapper:
        self.validate_can_generate_cpp_wrapper()
        self.wrapper_code = CudaWrapperCodeGen() if self.cuda else CppWrapperCodeGen()
        return
    device_types = self.device_types.copy()
    device_types.discard('cpu')
    assert len(device_types) <= 1, 'Does not support mixing {}'.format('+'.join(device_types))
    only_cpu = len(device_types) == 0
    device_type = 'cpu' if only_cpu else device_types.pop()
    wrapper_code_gen_cls = get_wrapper_codegen_for_device(device_type)
    assert wrapper_code_gen_cls is not None, f'Device {device_type} not supported'
    self.wrapper_code = wrapper_code_gen_cls()