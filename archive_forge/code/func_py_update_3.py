from __future__ import print_function, absolute_import, division
import unittest
import numpy as np
from numba import guvectorize
from numba.tests.support import TestCase
def py_update_3(x0_t, x1_t, x2_t, y_1):
    for t in range(0, x0_t.shape[0]):
        x0_t[t] = y_1[0]
        x1_t[t] = 2 * y_1[0]
        x2_t[t] = 3 * y_1[0]