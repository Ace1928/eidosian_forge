from Cython.Build.Dependencies import strip_string_literals
from Cython.TestUtils import CythonTest
def test_nested_quotes(self):
    self.t(' \'"\' "\'" ', ' \'_L1_\' "_L2_" ')