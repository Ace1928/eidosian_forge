import logging
import traceback
from typing import TYPE_CHECKING
import numpy as np
import torch
import onnxruntime as ort
from onnxruntime.capi.onnxruntime_inference_collection import OrtValue
from onnxruntime.transformers.io_binding_helper import TypeHelper as ORTTypeHelper
from ..utils import is_cupy_available, is_onnxruntime_training_available
@staticmethod
def prepare_io_binding(ort_model: 'ORTModel', **inputs) -> ort.IOBinding:
    """
        Returns an IOBinding object for an inference session. This method is for general purpose, if the inputs and outputs
        are determined, you can prepare data buffers directly to avoid tensor transfers across frameworks.
        """
    if not all((input_name in inputs.keys() for input_name in ort_model.inputs_names)):
        raise ValueError(f'The ONNX model takes {ort_model.inputs_names.keys()} as inputs, but only {inputs.keys()} are given.')
    name_to_np_type = TypeHelper.get_io_numpy_type_map(ort_model.model)
    io_binding = ort_model.model.io_binding()
    for input_name in ort_model.inputs_names:
        onnx_input = inputs.pop(input_name)
        onnx_input = onnx_input.contiguous()
        io_binding.bind_input(input_name, onnx_input.device.type, ort_model.device.index, name_to_np_type[input_name], list(onnx_input.size()), onnx_input.data_ptr())
    for name in ort_model.output_names:
        io_binding.bind_output(name, ort_model.device.type, device_id=ort_model.device.index)
    return io_binding