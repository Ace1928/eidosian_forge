import numpy as np
import ctypes
from numba.cuda.cudadrv.devicearray import (DeviceRecord, from_record_like,
from numba.cuda.testing import unittest, CUDATestCase
from numba.cuda.testing import skip_on_cudasim
from numba.np import numpy_support
from numba import cuda
def test_device_record_copy(self):
    hostrec = self.hostz.copy()
    devrec = DeviceRecord(self.dtype)
    devrec.copy_to_device(hostrec)
    hostrec2 = self.hostnz.copy()
    devrec.copy_to_host(hostrec2)
    np.testing.assert_equal(self.hostz, hostrec2)
    hostrec3 = self.hostnz.copy()
    devrec.copy_to_device(hostrec3)
    hostrec4 = self.hostz.copy()
    devrec.copy_to_host(hostrec4)
    np.testing.assert_equal(hostrec4, self.hostnz)