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
def generate_extern_kernel_alloc_and_find_schema_if_needed_fbcode(self, name, cpp_kernel_key, op_overload, raw_args, outputs):

    def extract_output_name(out):
        assert out is not None, 'None, i.e. optional output is not supported'
        if isinstance(out, ir.MultiOutput):
            return out.get_name()
        elif isinstance(out, (list, tuple)):
            return type(out)((extract_output_name(o) for o in out))
        else:
            raise AssertionError(f'Unexpected output: {type(out)}')
    output_args = extract_output_name(outputs)
    if isinstance(output_args, str):
        output_args = [output_args]
    tensor_call_args, int_call_args = self.generate_extern_kernel_args_decl_if_needed(op_overload, raw_args, output_args)
    tensor_call_args_str = ', '.join(tensor_call_args)
    int_call_args_str = ', '.join(int_call_args)
    extern_kernel_node_index = len(V.graph.extern_kernel_nodes) - 1
    self.writeline(f'aoti_torch_proxy_executor_call_function(proxy_executor, {extern_kernel_node_index}, {len(int_call_args)}, std::vector<int64_t>{{{int_call_args_str}}}.data(), {len(tensor_call_args)}, std::vector<AtenTensorHandle>{{{tensor_call_args_str}}}.data());')
    self.extern_call_ops.add(cpp_kernel_key)