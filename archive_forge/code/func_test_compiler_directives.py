import os
import tempfile
import unittest
from Cython.Shadow import inline
from Cython.Build.Inline import safe_type
from Cython.TestUtils import CythonTest
def test_compiler_directives(self):
    self.assertEqual(inline('return sum(x)', x=[1, 2, 3], cython_compiler_directives={'boundscheck': False}), 6)