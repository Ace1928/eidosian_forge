from __future__ import (  # for onnx.ModelProto (ONNXProgram) and onnxruntime (ONNXRuntimeOptions)
import abc
import contextlib
import dataclasses
import io
import logging
import os
import warnings
from collections import defaultdict
from typing import (
from typing_extensions import Self
import torch
import torch._ops
import torch.export as torch_export
import torch.utils._pytree as pytree
from torch._subclasses import fake_tensor
from torch.onnx._internal import _beartype, io_adapter
from torch.onnx._internal.diagnostics import infra
from torch.onnx._internal.fx import (
def common_pre_export_passes(options: ResolvedExportOptions, original_model: Union[torch.nn.Module, Callable], fx_module: torch.fx.GraphModule, fx_module_args: Sequence[Any]):
    from torch.onnx._internal.fx import analysis, passes
    diagnostic_context = options.diagnostic_context
    module = passes.Decompose(diagnostic_context, fx_module, options.decomposition_table, enable_dynamic_axes=options.dynamic_shapes, allow_fake_constant=options.fake_context is not None).run(*fx_module_args)
    module = passes.Functionalize(diagnostic_context, module, enable_dynamic_axes=options.dynamic_shapes, allow_fake_constant=options.fake_context is not None).run(*fx_module_args)
    module = passes.RemoveInputMutation(diagnostic_context, module).run(*fx_module_args)
    module = passes.InsertTypePromotion(diagnostic_context, module).run()
    analysis.UnsupportedFxNodesAnalysis(diagnostic_context, module, options.onnxfunction_dispatcher).analyze(infra.levels.ERROR)
    if isinstance(original_model, torch.nn.Module):
        module = passes.RestoreParameterAndBufferNames(diagnostic_context, module, original_model).run()
    module = passes.Modularize(diagnostic_context, module).run()
    options.fx_tracer.input_adapter.append_step(io_adapter.RemoveNoneInputStep())
    options.fx_tracer.input_adapter.append_step(io_adapter.RemoveNonTensorInputStep())
    options.fx_tracer.input_adapter.append_step(io_adapter.ConvertComplexToRealRepresentationInputStep())
    options.fx_tracer.output_adapter.append_step(io_adapter.FlattenOutputStep())
    options.fx_tracer.output_adapter.append_step(io_adapter.ConvertComplexToRealRepresentationOutputStep())
    return module