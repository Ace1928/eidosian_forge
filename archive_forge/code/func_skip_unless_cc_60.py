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
def skip_unless_cc_60(fn):
    return unittest.skipUnless(cc_X_or_above(6, 0), 'requires cc >= 6.0')(fn)