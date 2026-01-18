import numpy as np
from numba.core import config
from numba.cuda.testing import CUDATestCase
from numba import cuda
def test_creation_readout(self):
    for vty in vector_types.values():
        with self.subTest(vty=vty):
            arr = np.zeros((vty.num_elements,))
            kernel = make_kernel(vty)
            kernel[1, 1](arr)
            np.testing.assert_almost_equal(arr, np.array(range(vty.num_elements)))