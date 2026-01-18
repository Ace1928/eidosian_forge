from numba import njit, cfunc
from numba.tests.support import TestCase, unittest
from numba.core import cgutils
def make_testcase(self, src, fname):
    glb = {}
    exec(src, glb)
    fn = glb[fname]
    return fn