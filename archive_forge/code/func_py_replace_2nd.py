from __future__ import print_function, absolute_import, division
import unittest
import numpy as np
from numba import guvectorize
from numba.tests.support import TestCase
def py_replace_2nd(x_t, y_1):
    for t in range(0, x_t.shape[0], 2):
        x_t[t] = y_1[0]