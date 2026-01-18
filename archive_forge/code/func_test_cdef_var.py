from Cython.TestUtils import CythonTest
def test_cdef_var(self):
    self.t(u'\n                    cdef int hello\n                    cdef int hello = 4, x = 3, y, z\n                ')