import numpy as np
from numba import vectorize, guvectorize
from numba import cuda
from numba.cuda.cudadrv import driver
from numba.cuda.testing import unittest, ContextResettingTestCase, ForeignArray
from numba.cuda.testing import skip_on_cudasim, skip_if_external_memmgr
from numba.tests.support import linux_only, override_config
from unittest.mock import call, patch
def test_launch_no_sync(self):
    f_arr = ForeignArray(cuda.device_array(10))

    @cuda.jit
    def f(x):
        pass
    with patch.object(cuda.cudadrv.driver.Stream, 'synchronize', return_value=None) as mock_sync:
        f[1, 1](f_arr)
    mock_sync.assert_not_called()