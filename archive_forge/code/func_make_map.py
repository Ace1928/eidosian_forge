import collections.abc
import numbers
import struct
from cmath import isnan
from typing import (
import google.protobuf.message
import numpy as np
from onnx import (
def make_map(name: str, key_type: int, keys: List[Any], values: SequenceProto) -> MapProto:
    """Make a Map with specified key-value pair arguments.

    Criteria for conversion:
    - Keys and Values must have the same number of elements
    - Every key in keys must be of the same type
    - Every value in values must be of the same type
    """
    map_proto = MapProto()
    valid_key_int_types = [TensorProto.INT8, TensorProto.INT16, TensorProto.INT32, TensorProto.INT64, TensorProto.UINT8, TensorProto.UINT16, TensorProto.UINT32, TensorProto.UINT64]
    map_proto.name = name
    map_proto.key_type = key_type
    if key_type == TensorProto.STRING:
        map_proto.string_keys.extend(keys)
    elif key_type in valid_key_int_types:
        map_proto.keys.extend(keys)
    map_proto.values.CopyFrom(values)
    return map_proto