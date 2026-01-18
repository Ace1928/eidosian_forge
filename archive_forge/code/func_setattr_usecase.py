import unittest
from numba import jit
from numba.core import types
from numba.tests.support import TestCase
def setattr_usecase(o, v):
    o.x = v