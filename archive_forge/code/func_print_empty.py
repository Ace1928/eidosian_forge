import sys
import numpy as np
import unittest
from numba import jit, njit
from numba.core import types, errors, utils
from numba.tests.support import (captured_stdout, TestCase, EnableNRTStatsMixin)
def print_empty():
    print()