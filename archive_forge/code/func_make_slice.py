import numpy as np
from onnx.reference.op_run import OpRun
def make_slice(arr, axis, i):
    slc = [slice(None)] * arr.ndim
    slc[axis] = i
    return slc