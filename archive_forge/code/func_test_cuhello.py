from numba.cuda.testing import CUDATestCase, skip_on_cudasim
import subprocess
import sys
import unittest
from numba import cuda
from numba import cuda
from numba import cuda
from numba import cuda
from numba import cuda
import numpy as np
def test_cuhello(self):
    output, _ = self.run_code(cuhello_usecase)
    actual = [line.strip() for line in output.splitlines()]
    expected = ['-42'] * 6 + ['%d 999' % i for i in range(6)]
    self.assertEqual(sorted(actual), expected)