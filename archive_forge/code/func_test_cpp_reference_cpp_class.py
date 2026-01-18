import unittest
from Cython.Compiler import PyrexTypes as pt
from Cython.Compiler.ExprNodes import NameNode
from Cython.Compiler.PyrexTypes import CFuncTypeArg
def test_cpp_reference_cpp_class(self):
    classes = [cppclasstype('Test%d' % i, []) for i in range(2)]
    function_types = [cfunctype(pt.CReferenceType(classes[0])), cfunctype(pt.CReferenceType(classes[1]))]
    functions = [NameNode(None, type=t) for t in function_types]
    self.assertMatches(function_types[0], [classes[0]], functions)
    self.assertMatches(function_types[1], [classes[1]], functions)