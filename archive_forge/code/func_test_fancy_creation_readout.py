import numpy as np
from numba.core import config
from numba.cuda.testing import CUDATestCase
from numba import cuda
def test_fancy_creation_readout(self):
    for vty in vector_types.values():
        with self.subTest(vty=vty):
            kernel = make_fancy_creation_kernel(vty)
            expected = np.array([1, 1, 2, 3, 1, 3, 2, 1, 1, 1, 2, 3, 2, 3, 1, 2, 3, 1, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 1, 3, 1, 2, 1, 1, 1, 3, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 3, 1, 1, 2, 3, 4, 1, 2, 3, 4, 1, 1, 3, 4, 1, 2, 1, 4, 1, 2, 3, 1, 1, 1, 3, 4, 1, 2, 1, 4, 1, 2, 3, 1, 1, 1, 1, 4, 1, 1, 3, 1, 1, 2, 1, 1, 1, 1, 1, 4, 1, 1, 3, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 2, 3, 2, 3, 1, 3, 2, 3, 2, 1, 2, 3, 1, 1, 1, 2, 3, 3, 1, 2, 3, 3, 1, 2, 3, 1, 1, 2, 3, 1, 1, 4, 2, 3, 1, 4, 2, 3, 1, 1, 2, 3, 1, 1, 2, 3, 2, 3, 2, 3, 2, 3, 1, 4, 2, 3, 1, 1, 4, 2, 3, 1, 1, 2, 3, 1, 1, 2, 3, 4])
            arr = np.zeros(expected.shape)
            kernel[1, 1](arr)
            np.testing.assert_almost_equal(arr, expected)