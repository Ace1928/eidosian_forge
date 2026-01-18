from Cython.TestUtils import CythonTest
import Cython.Compiler.Errors as Errors
from Cython.Compiler.Nodes import *
from Cython.Compiler.ParseTreeTransforms import *
from Cython.Compiler.Buffer import *
def non_parse(self, expected_err, opts):
    self.parse_opts(opts, expect_error=True)
    self.assertEqual(expected_err, self.error.message_only)