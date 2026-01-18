import inspect
import numpy as np
import triton
import triton.language as tl
from .._C.libtriton.triton import interpreter as _interpreter
def set_grid_idx(self, x, y, z):
    assert x < self.grid_dim[0]
    assert y < self.grid_dim[1]
    assert z < self.grid_dim[2]
    self.grid_idx = (x, y, z)