from __future__ import annotations
from typing import ClassVar
import numpy as np
from onnx import TensorProto, subbyte
from onnx.helper import (
from onnx.reference.custom_element_types import (
from onnx.reference.op_run import OpRun
def get_zero_point_type(self, zero_point: np.ndarray) -> int:
    zero_point_type = None
    if zero_point.dtype == float8e4m3fn and zero_point.dtype.descr[0][0] == 'e4m3fn':
        zero_point_type = TensorProto.FLOAT8E4M3FN
    elif zero_point.dtype == float8e4m3fnuz and zero_point.dtype.descr[0][0] == 'e4m3fnuz':
        zero_point_type = TensorProto.FLOAT8E4M3FNUZ
    elif zero_point.dtype == float8e5m2 and zero_point.dtype.descr[0][0] == 'e5m2':
        zero_point_type = TensorProto.FLOAT8E5M2
    elif zero_point.dtype == float8e5m2fnuz and zero_point.dtype.descr[0][0] == 'e5m2fnuz':
        zero_point_type = TensorProto.FLOAT8E5M2FNUZ
    elif zero_point.dtype == uint4 and zero_point.dtype.descr[0][0] == 'uint4':
        zero_point_type = TensorProto.UINT4
    elif zero_point.dtype == int4 and zero_point.dtype.descr[0][0] == 'int4':
        zero_point_type = TensorProto.INT4
    else:
        zero_point_type = np_dtype_to_tensor_dtype(zero_point.dtype)
    return zero_point_type