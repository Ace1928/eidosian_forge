import numpy as np
import ctypes
from numba.cuda.cudadrv.devicearray import (DeviceRecord, from_record_like,
from numba.cuda.testing import unittest, CUDATestCase
from numba.cuda.testing import skip_on_cudasim
from numba.np import numpy_support
from numba import cuda
def test_auto_device(self):
    hostrec = self.hostnz.copy()
    devrec, new_gpu_obj = auto_device(hostrec)
    self._check_device_record(hostrec, devrec)
    self.assertTrue(new_gpu_obj)
    hostrec2 = self.hostz.copy()
    devrec.copy_to_host(hostrec2)
    np.testing.assert_equal(hostrec2, hostrec)