import sys
import unittest
from numba import njit
@njit
def f0_2(a, b):
    return a + b