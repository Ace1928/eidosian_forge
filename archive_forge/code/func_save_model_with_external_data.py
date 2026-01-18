from __future__ import annotations
import io
import os
from typing import Tuple, TYPE_CHECKING, Union
import torch
from torch.onnx import _type_utils as jit_type_utils
from torch.onnx._internal import _beartype
@_beartype.beartype
def save_model_with_external_data(basepath: str, model_location: str, initializer_location: str, torch_load_paths: Tuple[Union[str, io.BytesIO], ...], onnx_model: onnx.ModelProto, rename_initializer: bool=False) -> None:
    """Load PyTorch tensors from files and add to "onnx_model" as external initializers.

    Output files:
        ONNX model file path:
        ONNX initializer folder: os.path.join(basepath, initializer_location)

    After running this function, you can do
        ort_sess = onnxruntime.InferenceSession(os.path.join(basepath, model_location))
    to execute the model.

    Arguments:
        basepath: Base path of the external data file (e.g., "/tmp/large-onnx-model").
        model_location: Relative location of the ONNX model file.
            E.g., "model.onnx" so that the model file is saved to
            "/tmp/large-onnx-model/model.onnx".
        initializer_location: Relative location of the ONNX initializer folder.
            E.g., "initializers" so that the initializers are saved to
            "/tmp/large-onnx-model/initializers".
        torch_load_paths: Files which containing serialized PyTorch tensors to be saved
            as ONNX initializers. They are loaded by torch.load.
        onnx_model: ONNX model to be saved with external initializers.
            If an input name matches a tensor loaded from "torch_load_paths",
            the tensor will be saved as that input's external initializer.
        rename_initializer: Replaces "." by "_" for all ONNX initializer names.
            Not needed by the official torch.onnx.dynamo_export. This is a hack
            for supporting `FXSymbolicTracer` tracer with fake tensor mode.
            In short, `FXSymbolicTracer` lifts FX parameters (self.linear_weight)
            as inputs (`def forward(self, linear_weight)`) and therefore, `.` cannot be used.
    """
    import onnx
    onnx_model_with_initializers = onnx.ModelProto()
    onnx_model_with_initializers.CopyFrom(onnx_model)
    onnx_input_names = {input.name for input in onnx_model.graph.input}
    for path in torch_load_paths:
        state_dict = torch.load(path)
        for name, tensor in state_dict.items():
            if rename_initializer:
                name = name.replace('.', '_')
            if name in onnx_input_names:
                onnx_input_names.remove(name)
            else:
                for onnx_input_name in onnx_input_names:
                    if onnx_input_name.endswith(name) or name.endswith(onnx_input_name):
                        name = onnx_input_name
                        onnx_input_names.remove(onnx_input_name)
                        break
            relative_tensor_file_path = os.path.join(initializer_location, name)
            tensor_proto = _create_tensor_proto_with_external_data(tensor, name, relative_tensor_file_path, basepath)
            onnx_model_with_initializers.graph.initializer.append(tensor_proto)
    onnx.save(onnx_model_with_initializers, os.path.join(basepath, model_location))