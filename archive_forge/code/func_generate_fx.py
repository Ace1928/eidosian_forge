from __future__ import annotations
from typing import Any, Callable, Mapping, Optional, Sequence, Union
import torch._dynamo
import torch.fx
import torch.onnx
from torch.onnx._internal import _beartype, exporter, io_adapter
from torch.onnx._internal.diagnostics import infra
def generate_fx(self, options: exporter.ResolvedExportOptions, model: 'ExportedProgram', model_args: Sequence[Any], model_kwargs: Mapping[str, Any]) -> torch.fx.GraphModule:
    self.input_adapter.append_step(io_adapter.FlattenInputWithTreeSpecValidationInputStep())
    self.input_adapter.append_step(io_adapter.PrependParamsBuffersConstantAotAutogradInputStep())
    options.fx_tracer.input_adapter.append_step(io_adapter.RemoveNoneInputStep())
    options.fx_tracer.input_adapter.append_step(io_adapter.RemoveNonTensorInputStep())
    options.fx_tracer.input_adapter.append_step(io_adapter.ConvertComplexToRealRepresentationInputStep())
    updated_model_args = self.input_adapter.apply(*model_args, model=model, **model_kwargs)
    options.fx_tracer.output_adapter.append_step(io_adapter.FlattenOutputStep())
    options.fx_tracer.output_adapter.append_step(io_adapter.PrependParamsAndBuffersAotAutogradOutputStep())
    model = model.run_decompositions(options.decomposition_table)
    return self.pre_export_passes(options, model, model.graph_module, updated_model_args)