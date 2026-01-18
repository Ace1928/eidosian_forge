from __future__ import print_function
import numpy as np
from numba import config, cuda, int32
from numba.cuda.testing import (unittest, CUDATestCase, skip_on_cudasim,
@skip_unless_cc_60
def test_max_cooperative_grid_blocks(self):
    sig = (int32[:, ::1],)
    c_sequential_rows = cuda.jit(sig)(sequential_rows)
    overload = c_sequential_rows.overloads[sig]
    blocks1d = overload.max_cooperative_grid_blocks(256)
    blocks2d = overload.max_cooperative_grid_blocks((16, 16))
    blocks3d = overload.max_cooperative_grid_blocks((16, 4, 4))
    self.assertEqual(blocks1d, blocks2d)
    self.assertEqual(blocks1d, blocks3d)