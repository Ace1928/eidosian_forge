import collections.abc
import numbers
import struct
from cmath import isnan
from typing import (
import google.protobuf.message
import numpy as np
from onnx import (
def set_metadata_props(proto: Union[ModelProto, GraphProto, FunctionProto, NodeProto, TensorProto, ValueInfoProto], dict_value: Dict[str, str]) -> None:
    del proto.metadata_props[:]
    for k, v in dict_value.items():
        entry = proto.metadata_props.add()
        entry.key = k
        entry.value = v