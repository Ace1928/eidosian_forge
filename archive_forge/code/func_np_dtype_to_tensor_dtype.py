import collections.abc
import numbers
import struct
from cmath import isnan
from typing import (
import google.protobuf.message
import numpy as np
from onnx import (
def np_dtype_to_tensor_dtype(np_dtype: np.dtype) -> int:
    """Convert a numpy's dtype to corresponding tensor type. It can be used while converting numpy arrays to tensors.

    Args:
        np_dtype: numpy's data_type

    Returns:
        TensorsProto's data_type
    """
    return cast(int, mapping._NP_TYPE_TO_TENSOR_TYPE[np_dtype])