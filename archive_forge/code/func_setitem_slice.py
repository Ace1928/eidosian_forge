import numpy as np
import unittest
from numba import njit
from numba.core import types, errors
from numba.tests.support import TestCase
def setitem_slice(a, start, stop, step, scalar):
    a[start:stop:step] = scalar