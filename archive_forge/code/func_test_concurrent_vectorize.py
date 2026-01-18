import threading
import random
import numpy as np
from numba import jit, vectorize, guvectorize
from numba.tests.support import temp_directory, override_config
from numba.core import config
import unittest
def test_concurrent_vectorize(self):
    self.run_compile([self.run_vectorize(nopython=True)])