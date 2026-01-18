import inspect
import numpy as np
import triton
import triton.language as tl
from .._C.libtriton.triton import interpreter as _interpreter
def create_dot(self, a, b, d, allow_tf32, maxNumImpreciseAcc):
    return TensorHandle(np.dot(a.data, b.data) + d.data, a.dtype)