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
def skip_on_cudasim(reason):
    """Skip this test if running on the CUDA simulator"""
    return unittest.skipIf(config.ENABLE_CUDASIM, reason)