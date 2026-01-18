import os
import re
import sys
import uuid
from itertools import chain
from typing import Callable, Iterable, Optional
import onnx.onnx_cpp2py_export.checker as c_checker
from onnx.onnx_pb import AttributeProto, GraphProto, ModelProto, TensorProto
def save_external_data(tensor: TensorProto, base_path: str) -> None:
    """Writes tensor data to an external file according to information in the `external_data` field.

    Arguments:
        tensor (TensorProto): Tensor object to be serialized
        base_path: System path of a folder where tensor data is to be stored
    """
    info = ExternalDataInfo(tensor)
    external_data_file_path = os.path.join(base_path, info.location)
    if not tensor.HasField('raw_data'):
        raise ValueError("raw_data field doesn't exist.")
    if not os.path.isfile(external_data_file_path):
        with open(external_data_file_path, 'ab'):
            pass
    with open(external_data_file_path, 'r+b') as data_file:
        data_file.seek(0, 2)
        if info.offset is not None:
            file_size = data_file.tell()
            if info.offset > file_size:
                data_file.write(b'\x00' * (info.offset - file_size))
            data_file.seek(info.offset)
        offset = data_file.tell()
        data_file.write(tensor.raw_data)
        set_external_data(tensor, info.location, offset, data_file.tell() - offset)