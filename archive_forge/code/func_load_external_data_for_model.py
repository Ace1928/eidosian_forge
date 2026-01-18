import os
import re
import sys
import uuid
from itertools import chain
from typing import Callable, Iterable, Optional
import onnx.onnx_cpp2py_export.checker as c_checker
from onnx.onnx_pb import AttributeProto, GraphProto, ModelProto, TensorProto
def load_external_data_for_model(model: ModelProto, base_dir: str) -> None:
    """Loads external tensors into model

    Arguments:
        model: ModelProto to load external data to
        base_dir: directory that contains external data
    """
    for tensor in _get_all_tensors(model):
        if uses_external_data(tensor):
            load_external_data_for_tensor(tensor, base_dir)
            tensor.data_location = TensorProto.DEFAULT
            del tensor.external_data[:]