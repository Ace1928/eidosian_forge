import collections.abc
import numbers
import struct
from cmath import isnan
from typing import (
import google.protobuf.message
import numpy as np
from onnx import (
def make_empty_tensor_value_info(name: str) -> ValueInfoProto:
    value_info_proto = ValueInfoProto()
    value_info_proto.name = name
    return value_info_proto