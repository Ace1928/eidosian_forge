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
def ort_type_to_numpy_type(ort_type: str):
    ort_type_to_numpy_type_map = {'tensor(int64)': np.int64, 'tensor(int32)': np.int32, 'tensor(int8)': np.int8, 'tensor(float)': np.float32, 'tensor(float16)': np.float16, 'tensor(bool)': bool}
    if ort_type in ort_type_to_numpy_type_map:
        return ort_type_to_numpy_type_map[ort_type]
    else:
        raise ValueError(f'{ort_type} is not supported. Here is a list of supported data type: {ort_type_to_numpy_type_map.keys()}')