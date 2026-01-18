from __future__ import print_function
import numpy as np
from numba import config, cuda, int32
from numba.cuda.testing import (unittest, CUDATestCase, skip_on_cudasim,
@skip_on_cudasim('Simulator does not implement linking')
def test_false_cooperative_doesnt_link_cudadevrt(self):
    """
        We should only mark a kernel as cooperative and link cudadevrt if the
        kernel uses grid sync. Here we ensure that one that doesn't use grid
        synsync isn't marked as such.
        """
    A = np.full(1, fill_value=np.nan)
    no_sync[1, 1](A)
    for key, overload in no_sync.overloads.items():
        self.assertFalse(overload.cooperative)
        for link in overload._codelibrary._linking_files:
            self.assertNotIn('cudadevrt', link)