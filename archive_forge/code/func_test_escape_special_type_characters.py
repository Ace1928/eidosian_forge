from __future__ import absolute_import
import unittest
import Cython.Compiler.PyrexTypes as PT
def test_escape_special_type_characters(self):
    test_func = PT._escape_special_type_characters
    function_name = '_escape_special_type_characters'
    self._test_escape(function_name)