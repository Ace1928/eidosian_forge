import numpy as np
from numba import vectorize, guvectorize
from numba import cuda
from numba.cuda.cudadrv import driver
from numba.cuda.testing import unittest, ContextResettingTestCase, ForeignArray
from numba.cuda.testing import skip_on_cudasim, skip_if_external_memmgr
from numba.tests.support import linux_only, override_config
from unittest.mock import call, patch
def test_consume_sync_disabled(self):
    s = cuda.stream()
    f_arr = ForeignArray(cuda.device_array(10, stream=s))
    with override_config('CUDA_ARRAY_INTERFACE_SYNC', False):
        with patch.object(cuda.cudadrv.driver.Stream, 'synchronize', return_value=None) as mock_sync:
            cuda.as_cuda_array(f_arr)
        mock_sync.assert_not_called()