import numpy as np
import ctypes
from numba.cuda.cudadrv.devicearray import (DeviceRecord, from_record_like,
from numba.cuda.testing import unittest, CUDATestCase
from numba.cuda.testing import skip_on_cudasim
from numba.np import numpy_support
from numba import cuda
def test_device_record_interface(self):
    hostrec = self.hostz.copy()
    devrec = DeviceRecord(self.dtype)
    self._check_device_record(hostrec, devrec)