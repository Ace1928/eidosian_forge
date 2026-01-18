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
def generate_return(self, output_refs):
    if V.graph.aot_mode:
        cst_names = V.graph.constants.keys()
        for idx, output in enumerate(output_refs):
            if output in cst_names:
                if config.aot_inductor.abi_compatible:
                    self.wrapper_call.writeline(f'aoti_torch_clone({output}, &output_handles[{idx}]);')
                else:
                    self.wrapper_call.writeline(f'output_handles[{idx}] = reinterpret_cast<AtenTensorHandle>(' + f'new at::Tensor(std::move({output}.clone())));')
            elif config.aot_inductor.abi_compatible:
                if output in self.cached_thread_locals:
                    self.wrapper_call.writeline(f'aoti_torch_new_uninitialized_tensor(&output_handles[{idx}]);')
                    self.wrapper_call.writeline(f'aoti_torch_assign_tensors({output}, output_handles[{idx}]);')
                else:
                    self.wrapper_call.writeline(f'output_handles[{idx}] = {output}.release();')
            else:
                self.wrapper_call.writeline(f'output_handles[{idx}] = reinterpret_cast<AtenTensorHandle>(' + f'new at::Tensor({output}));')
    else:
        self.wrapper_call.writeline(f'return {{{', '.join(output_refs)}}};\n}}')