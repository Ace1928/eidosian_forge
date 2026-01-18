import os
import platform
import shutil
from numba.tests.support import SerialMixin
from numba.cuda.cuda_paths import get_conda_ctk
from numba.cuda.cudadrv import driver, devices, libs
from numba.core import config
from numba.tests.support import TestCase
from pathlib import Path
import unittest
def skip_if_cuda_includes_missing(fn):
    cuda_h = os.path.join(config.CUDA_INCLUDE_PATH, 'cuda.h')
    cuda_h_file = os.path.exists(cuda_h) and os.path.isfile(cuda_h)
    reason = 'CUDA include dir not available on this system'
    return unittest.skipUnless(cuda_h_file, reason)(fn)