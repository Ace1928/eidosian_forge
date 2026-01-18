import numpy as np
from onnx.helper import tensor_dtype_to_np_dtype
from onnx.reference.op_run import OpRun
@staticmethod
def numpy_type(dtype):
    return tensor_dtype_to_np_dtype(dtype)