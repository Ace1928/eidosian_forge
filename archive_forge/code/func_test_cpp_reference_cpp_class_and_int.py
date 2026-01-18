import unittest
from Cython.Compiler import PyrexTypes as pt
from Cython.Compiler.ExprNodes import NameNode
from Cython.Compiler.PyrexTypes import CFuncTypeArg
def test_cpp_reference_cpp_class_and_int(self):
    classes = [cppclasstype('Test%d' % i, []) for i in range(2)]
    function_types = [cfunctype(pt.CReferenceType(classes[0]), pt.c_int_type), cfunctype(pt.CReferenceType(classes[0]), pt.c_long_type), cfunctype(pt.CReferenceType(classes[1]), pt.c_int_type), cfunctype(pt.CReferenceType(classes[1]), pt.c_long_type)]
    functions = [NameNode(None, type=t) for t in function_types]
    self.assertMatches(function_types[0], [classes[0], pt.c_int_type], functions)
    self.assertMatches(function_types[1], [classes[0], pt.c_long_type], functions)
    self.assertMatches(function_types[2], [classes[1], pt.c_int_type], functions)
    self.assertMatches(function_types[3], [classes[1], pt.c_long_type], functions)