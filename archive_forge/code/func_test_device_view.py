import ctypes
import numpy as np
from numba.cuda.cudadrv import driver, drvapi, devices
from numba.cuda.testing import unittest, ContextResettingTestCase
from numba.cuda.testing import skip_on_cudasim
def test_device_view(self):
    devmem = self.context.memalloc(1024)
    self._template(devmem.view(10))