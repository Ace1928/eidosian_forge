import inspect
import numpy as np
import triton
import triton.language as tl
from .._C.libtriton.triton import interpreter as _interpreter
def get_null_value(self, type):
    return TensorHandle(np.array([0], dtype=self.np_dtype(type)), type)