from Cython.TestUtils import CythonTest
import Cython.Compiler.Errors as Errors
from Cython.Compiler.Nodes import *
from Cython.Compiler.ParseTreeTransforms import *
from Cython.Compiler.Buffer import *
def test_general_slice(self):
    self.parse(u'cdef float[::ptr, ::direct & contig, 0::full & strided] x')