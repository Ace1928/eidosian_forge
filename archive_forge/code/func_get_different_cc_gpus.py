import multiprocessing
import os
import shutil
import unittest
import warnings
from numba import cuda
from numba.core.errors import NumbaWarning
from numba.cuda.testing import (CUDATestCase, skip_on_cudasim,
from numba.tests.support import SerialMixin
from numba.tests.test_caching import (DispatcherCacheUsecasesTest,
def get_different_cc_gpus():
    first_gpu = cuda.gpus[0]
    with first_gpu:
        first_cc = cuda.current_context().device.compute_capability
    for gpu in cuda.gpus[1:]:
        with gpu:
            cc = cuda.current_context().device.compute_capability
            if cc != first_cc:
                return (first_gpu, gpu)
    return None