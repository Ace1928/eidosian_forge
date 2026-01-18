import collections.abc
import numbers
import struct
from cmath import isnan
from typing import (
import google.protobuf.message
import numpy as np
from onnx import (
def printable_value_info(v: ValueInfoProto) -> str:
    s = f'%{v.name}'
    if v.type:
        s = f'{s}[{printable_type(v.type)}]'
    return s