import os
import re
import sys
import uuid
from itertools import chain
from typing import Callable, Iterable, Optional
import onnx.onnx_cpp2py_export.checker as c_checker
from onnx.onnx_pb import AttributeProto, GraphProto, ModelProto, TensorProto
def load_external_data_for_tensor(tensor: TensorProto, base_dir: str) -> None:
    """Loads data from an external file for tensor.
    Ideally TensorProto should not hold any raw data but if it does it will be ignored.

    Arguments:
        tensor: a TensorProto object.
        base_dir: directory that contains the external data.
    """
    info = ExternalDataInfo(tensor)
    external_data_file_path = c_checker._resolve_external_data_location(base_dir, info.location, tensor.name)
    with open(external_data_file_path, 'rb') as data_file:
        if info.offset:
            data_file.seek(info.offset)
        if info.length:
            tensor.raw_data = data_file.read(info.length)
        else:
            tensor.raw_data = data_file.read()