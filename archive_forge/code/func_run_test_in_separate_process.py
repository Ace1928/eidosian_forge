import os
import sys
import subprocess
import threading
from numba import cuda
from numba.cuda.testing import (unittest, CUDATestCase, skip_on_cudasim,
from numba.tests.support import captured_stdout
def run_test_in_separate_process(self, envvar, envvar_value):
    env_copy = os.environ.copy()
    env_copy[envvar] = str(envvar_value)
    code = "if 1:\n            from numba import cuda\n            @cuda.jit('(int64,)')\n            def kernel(x):\n                pass\n            kernel(1,)\n            "
    cmdline = [sys.executable, '-c', code]
    return self.run_cmd(cmdline, env_copy)