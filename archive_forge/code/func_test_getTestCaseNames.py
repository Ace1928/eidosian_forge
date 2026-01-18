import functools
import sys
import types
import warnings
import unittest
def test_getTestCaseNames(self):
    with self.assertWarns(DeprecationWarning) as w:
        tests = unittest.getTestCaseNames(self.MyTestCase, prefix='check', sortUsing=self.reverse_three_way_cmp, testNamePatterns=None)
    self.assertEqual(w.filename, __file__)
    self.assertEqual(tests, ['check_2', 'check_1'])