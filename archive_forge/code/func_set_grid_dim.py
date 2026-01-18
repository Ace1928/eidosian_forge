import inspect
import numpy as np
import triton
import triton.language as tl
from .._C.libtriton.triton import interpreter as _interpreter
def set_grid_dim(self, nx, ny, nz):
    self.grid_dim = (nx, ny, nz)