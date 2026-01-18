from itertools import product
import numpy as np
from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
from unittest.mock import patch
@skip_on_cudasim('cudasim does not use streams and operates synchronously')
def test_no_sync_default_stream(self):
    streams = (cuda.stream(), cuda.default_stream(), cuda.legacy_default_stream(), cuda.per_thread_default_stream())
    for stream in streams:
        darr = cuda.to_device(np.arange(5), stream=stream)
        with patch.object(cuda.cudadrv.driver.Stream, 'synchronize', return_value=None) as mock_sync:
            darr[0] = 10
        mock_sync.assert_not_called()