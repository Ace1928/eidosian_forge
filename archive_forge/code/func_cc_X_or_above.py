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
def cc_X_or_above(major, minor):
    if not config.ENABLE_CUDASIM:
        cc = devices.get_context().device.compute_capability
        return cc >= (major, minor)
    else:
        return True