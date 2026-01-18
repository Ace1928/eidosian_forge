import numpy as np
from numba import jit, njit
from numba.core import types
from numba.tests.support import TestCase, tag
import unittest
def dobool(a):
    return bool(a)