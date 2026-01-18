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
def generate_end(self, result):
    if V.graph.aot_mode:
        result.writeline('} // AOTInductorModel::run_impl')
        result.writeline('} // namespace aot_inductor')
        result.writeline('} // namespace torch')
        return
    result.writeline("'''\n)")
    wrapper_call_hash = codecache.code_hash(result.getvalue())
    result.splice(f"\n            module = CppWrapperCodeCache.load(cpp_wrapper_src, '{self.call_func_name}', '{wrapper_call_hash}', {self.cuda})\n            ")
    if all((x for x in self.output_is_tensor.values())):
        return_str = 'return f(args_tensor)'
    else:
        outputs = [f'outputs[{i}]' if self.output_is_tensor[i] else f'outputs[{i}].item()' for i in range(len(V.graph.graph_outputs))]
        outputs_str = f'[{', '.join(outputs)}]'
        return_str = f'\n                    outputs = f(args_tensor)\n                    return {outputs_str}\n            '
    args_str = 'args_tensor = [arg if isinstance(arg, torch.Tensor) else torch.tensor(arg) for arg in args]'
    if V.graph.constants:
        assert all((isinstance(v, torch.Tensor) for v in list(V.graph.constants.values()))), 'Expect all constants to be Tensor'
        constants_str = f'[{', '.join(V.graph.constants.keys())}]'
        args_str += f'\n                    constants_tensor = {constants_str}\n                    args_tensor.extend(constants_tensor)\n            '
    result.splice(f'\n            def _wrap_func(f):\n                def g(args):\n                    {args_str}\n                    {return_str}\n                return g\n            call = _wrap_func(module.{self.call_func_name})\n            ')