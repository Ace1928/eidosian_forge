import numpy as np
from numba import jit
from numba.core import types
from numba.tests.support import TestCase, tag
import unittest
def array_overlap2(src, dest, k=1):
    assert src.shape == dest.shape
    dest[:-k] = src[k:]