from Cython.TestUtils import CythonTest
import Cython.Compiler.Errors as Errors
from Cython.Compiler.Nodes import *
from Cython.Compiler.ParseTreeTransforms import *
from Cython.Compiler.Buffer import *
def not_parseable(self, expected_error, s):
    e = self.should_fail(lambda: self.fragment(s), Errors.CompileError)
    self.assertEqual(expected_error, e.message_only)