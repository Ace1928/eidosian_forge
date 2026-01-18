from Cython.TestUtils import CythonTest
import Cython.Compiler.Errors as Errors
from Cython.Compiler.Nodes import *
from Cython.Compiler.ParseTreeTransforms import *
from Cython.Compiler.Buffer import *
def disable_test_no_buf_arg(self):
    self.not_parseable(u"Expected ']'", u'cdef extern foo(object[int, ndim=2])')