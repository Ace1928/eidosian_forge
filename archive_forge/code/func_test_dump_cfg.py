import numpy as np
from numba.core.utils import PYVERSION
from numba.cuda.testing import skip_on_cudasim, CUDATestCase
from numba.tests.support import (override_config, captured_stderr,
from numba import cuda, float64
import unittest
def test_dump_cfg(self):
    with override_config('DUMP_CFG', True):
        out = self.compile_simple_cuda()
    self.check_debug_output(out, ['cfg'])