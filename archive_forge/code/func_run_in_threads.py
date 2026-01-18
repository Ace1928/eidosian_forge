import ctypes
import ctypes.util
import os
import sys
import threading
import warnings
import numpy as np
import unittest
from numba import jit
from numba.core import errors
from numba.tests.support import TestCase, tag
def run_in_threads(self, func, n_threads):
    threads = []
    func(self.make_test_array(1), np.arange(1, dtype=np.intp))
    arr = self.make_test_array(50)
    for i in range(n_threads):
        indices = np.arange(arr.size, dtype=np.intp)
        np.random.shuffle(indices)
        t = threading.Thread(target=func, args=(arr, indices))
        threads.append(t)
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return arr