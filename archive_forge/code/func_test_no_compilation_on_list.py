import os
import sys
import subprocess
from numba import cuda
import unittest
import itertools
def test_no_compilation_on_list(self):
    code = 'if 1:\n        from unittest import mock\n        from llvmlite import binding as llvm\n        error = RuntimeError("Detected compilation during test listing")\n        with mock.patch.object(llvm.ExecutionEngine, \'finalize_object\',\n                               side_effect=error):\n            import numba\n            {0}\n        '
    with self.assertRaises(subprocess.CalledProcessError) as raises:
        cmd = [sys.executable, '-c', code.format('numba.njit(lambda:0)()')]
        subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=60)
    self.assertIn('Detected compilation during test listing', raises.exception.stdout.decode('UTF-8'))
    cmd = [sys.executable, '-c', code.format("numba.test('-l')")]
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)