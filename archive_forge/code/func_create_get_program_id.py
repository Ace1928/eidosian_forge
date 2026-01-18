import inspect
import numpy as np
import triton
import triton.language as tl
from .._C.libtriton.triton import interpreter as _interpreter
def create_get_program_id(self, axis):
    assert self.grid_idx is not None
    return TensorHandle(np.array([self.grid_idx[axis]], dtype=np.int32), tl.int32)