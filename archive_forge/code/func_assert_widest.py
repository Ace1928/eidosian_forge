from __future__ import absolute_import
import unittest
import Cython.Compiler.PyrexTypes as PT
def assert_widest(type1, type2, widest):
    self.assertEqual(widest, PT.widest_numeric_type(type1, type2))