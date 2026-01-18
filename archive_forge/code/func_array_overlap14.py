import numpy as np
from numba import jit
from numba.core import types
from numba.tests.support import TestCase, tag
import unittest
def array_overlap14(src, dest):
    assert src.shape == dest.shape
    dest[:] = src[:, ::-1]