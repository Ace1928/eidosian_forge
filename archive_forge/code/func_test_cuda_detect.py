import os
import sys
import subprocess
import threading
from numba import cuda
from numba.cuda.testing import (unittest, CUDATestCase, skip_on_cudasim,
from numba.tests.support import captured_stdout
def test_cuda_detect(self):
    with captured_stdout() as out:
        cuda.detect()
    output = out.getvalue()
    self.assertIn('Found', output)
    self.assertIn('CUDA devices', output)