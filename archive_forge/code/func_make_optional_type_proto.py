import collections.abc
import numbers
import struct
from cmath import isnan
from typing import (
import google.protobuf.message
import numpy as np
from onnx import (
def make_optional_type_proto(inner_type_proto: TypeProto) -> TypeProto:
    """Makes an optional TypeProto."""
    type_proto = TypeProto()
    type_proto.optional_type.elem_type.CopyFrom(inner_type_proto)
    return type_proto