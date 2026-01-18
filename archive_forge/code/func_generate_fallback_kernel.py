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
def generate_fallback_kernel(self, fallback_kernel, args):
    if V.graph.aot_mode and config.aot_inductor.abi_compatible:
        self.generate_c_shim_fallback_kernel(fallback_kernel, args)
    else:
        super().generate_fallback_kernel(fallback_kernel, args)