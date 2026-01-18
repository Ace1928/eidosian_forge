import unittest
from Cython.Compiler import PyrexTypes as pt
from Cython.Compiler.ExprNodes import NameNode
from Cython.Compiler.PyrexTypes import CFuncTypeArg
def test_cpp_reference_single_arg(self):
    function_types = [cfunctype(pt.CReferenceType(pt.c_int_type)), cfunctype(pt.CReferenceType(pt.c_long_type)), cfunctype(pt.CReferenceType(pt.c_double_type))]
    functions = [NameNode(None, type=t) for t in function_types]
    self.assertMatches(function_types[0], [pt.c_int_type], functions)
    self.assertMatches(function_types[1], [pt.c_long_type], functions)
    self.assertMatches(function_types[2], [pt.c_double_type], functions)