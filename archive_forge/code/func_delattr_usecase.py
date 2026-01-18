import unittest
from numba import jit
from numba.core import types
from numba.tests.support import TestCase
def delattr_usecase(o):
    del o.x