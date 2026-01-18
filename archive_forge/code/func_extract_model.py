from __future__ import annotations
import os
import onnx.checker
import onnx.helper
import onnx.shape_inference
from onnx import FunctionProto, ModelProto, NodeProto, TensorProto, ValueInfoProto
def extract_model(self, input_names: list[str], output_names: list[str]) -> ModelProto:
    inputs = self._collect_new_inputs(input_names)
    outputs = self._collect_new_outputs(output_names)
    nodes = self._collect_reachable_nodes(input_names, output_names)
    initializer, value_info = self._collect_reachable_tensors(nodes)
    local_functions = self._collect_referred_local_functions(nodes)
    model = self._make_model(nodes, inputs, outputs, initializer, value_info, local_functions)
    return model