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
def codegen_model_constructor(self):
    """
        // Generated code example
        AOTInductorModel::AOTInductorModel()
            : AOTInductorModelBase(4, 1) {
        inputs_info_[0].name = "input0";
        inputs_info_[0].dtype = "torch.float16";
        ...
        constants_info_[0].name = "L__self___weight";
        constants_info_[0].dtype = at::kFloat;
        constants_info_[0].offset = 0;
        constants_info_[0].data_size = 8192;
        constants_info_[0].shape = {64, 32};
        constants_info_[0].stride = {32, 1};
        ...
        outputs_info_[0].name = "output0";
        outputs_info_[0].dtype = "torch.float16";
        }
        """
    num_inputs = len(V.graph.graph_inputs)
    num_outputs = len(V.graph.graph_outputs)
    num_constants = len(V.graph.constants)
    self.prefix.splice(f'\n            AOTInductorModel::AOTInductorModel(std::shared_ptr<ConstantMap> constants_map, std::optional<std::string> cubin_dir)\n                : AOTInductorModelBase({num_inputs}, {num_outputs}, {num_constants}, cubin_dir) {{\n            ')
    with self.prefix.indent():
        for idx, (name, inp) in enumerate(V.graph.graph_inputs.items()):
            assert not isinstance(inp, sympy.Expr), f'input name={name!r} cannot be symbolic'
            self.write_input_output_info('inputs_info_', idx, name)
        for idx, (name, tensor) in enumerate(V.graph.constants.items()):
            assert isinstance(tensor, torch.Tensor)
            self.prefix.writeline(f'constants_info_[{idx}].name = "{name}";')
            self.prefix.writeline(f'constants_info_[{idx}].dtype = static_cast<int32_t>({self.codegen_dtype(tensor.dtype)});')
            self.prefix.writeline(f'constants_info_[{idx}].offset = {tensor.storage_offset()};')
            self.prefix.writeline(f'constants_info_[{idx}].data_size = {tensor.untyped_storage().nbytes()};')
            size_str = ', '.join([str(s) for s in tensor.size()])
            self.prefix.writeline(f'constants_info_[{idx}].shape = {{{size_str}}};')
            stride_str = ', '.join([str(s) for s in tensor.stride()])
            self.prefix.writeline(f'constants_info_[{idx}].stride = {{{stride_str}}};')
        self.prefix.writeline('update_constants_map(std::move(constants_map));')

        def escape_string(x):
            return x.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n').replace('\t', '\\t')
        self.prefix.writeline(f'in_spec_ = "{escape_string(config.aot_inductor.serialized_in_spec)}";')
        self.prefix.writeline(f'out_spec_ = "{escape_string(config.aot_inductor.serialized_out_spec)}";')
        for idx, output in enumerate(V.graph.graph_outputs):
            assert not isinstance(output, sympy.Expr), f'output name={name!r} cannot be symbolic'
            name = f'output{idx}'
            self.write_input_output_info('outputs_info_', idx, name)
        self.prefix.writeline('this->kernels_ = std::make_unique<AOTInductorModelKernels>();')
    self.prefix.writeline('}')