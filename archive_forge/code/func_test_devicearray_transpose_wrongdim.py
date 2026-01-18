import itertools
import numpy as np
from numba.cuda.cudadrv import devicearray
from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase
from numba.cuda.testing import skip_on_cudasim
@skip_on_cudasim('This works in the simulator')
def test_devicearray_transpose_wrongdim(self):
    gpumem = cuda.to_device(np.array(np.arange(12)).reshape(3, 4, 1))
    with self.assertRaises(NotImplementedError) as e:
        np.transpose(gpumem)
    self.assertEqual("transposing a non-2D DeviceNDArray isn't supported", str(e.exception))