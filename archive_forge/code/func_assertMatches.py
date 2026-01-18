import unittest
from Cython.Compiler import PyrexTypes as pt
from Cython.Compiler.ExprNodes import NameNode
from Cython.Compiler.PyrexTypes import CFuncTypeArg
def assertMatches(self, expected_type, arg_types, functions):
    match = pt.best_match(arg_types, functions)
    if expected_type is not None:
        self.assertNotEqual(None, match)
    self.assertEqual(expected_type, match.type)