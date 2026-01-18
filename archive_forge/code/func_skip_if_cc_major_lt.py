import numpy as np
from ctypes import byref, c_size_t
from numba.cuda.cudadrv.driver import device_memset, driver, USE_NV_BINDING
from numba import cuda
from numba.cuda.testing import unittest, ContextResettingTestCase
from numba.cuda.testing import skip_on_cudasim, skip_on_arm
from numba.tests.support import linux_only
def skip_if_cc_major_lt(self, min_required, reason):
    """
        Skip the current test if the compute capability of the device is
        less than `min_required`.
        """
    ctx = cuda.current_context()
    cc_major = ctx.device.compute_capability[0]
    if cc_major < min_required:
        self.skipTest(reason)