from Cython.Build.Dependencies import strip_string_literals
from Cython.TestUtils import CythonTest
def test_extern(self):
    self.t("cdef extern from 'a.h': # comment", "cdef extern from '_L1_': #_L2_")