import os
import re
import sys
import uuid
from itertools import chain
from typing import Callable, Iterable, Optional
import onnx.onnx_cpp2py_export.checker as c_checker
from onnx.onnx_pb import AttributeProto, GraphProto, ModelProto, TensorProto
def set_external_data(tensor: TensorProto, location: str, offset: Optional[int]=None, length: Optional[int]=None, checksum: Optional[str]=None, basepath: Optional[str]=None) -> None:
    if not tensor.HasField('raw_data'):
        raise ValueError('Tensor ' + tensor.name + 'does not have raw_data field. Cannot set external data for this tensor.')
    del tensor.external_data[:]
    tensor.data_location = TensorProto.EXTERNAL
    for k, v in {'location': location, 'offset': int(offset) if offset is not None else None, 'length': int(length) if length is not None else None, 'checksum': checksum, 'basepath': basepath}.items():
        if v is not None:
            entry = tensor.external_data.add()
            entry.key = k
            entry.value = str(v)