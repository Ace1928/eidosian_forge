import collections.abc
import numbers
import struct
from cmath import isnan
from typing import (
import google.protobuf.message
import numpy as np
from onnx import (
def tensor_dtype_to_storage_tensor_dtype(tensor_dtype: int) -> int:
    """Convert a TensorProto's data_type to corresponding data_type for storage.

    Args:
        tensor_dtype: TensorProto's data_type

    Returns:
        data_type for storage
    """
    return mapping.TENSOR_TYPE_MAP[tensor_dtype].storage_dtype