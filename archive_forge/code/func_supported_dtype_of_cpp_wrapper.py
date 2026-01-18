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
def supported_dtype_of_cpp_wrapper(dtype, cuda):
    supported_dtype = {torch.float32, torch.float64, torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8, torch.bool, torch.bfloat16, torch.complex64}
    if cuda:
        supported_dtype.add(torch.float16)
        supported_dtype.add(torch.float8_e4m3fn)
        supported_dtype.add(torch.float8_e5m2)
    return dtype in supported_dtype