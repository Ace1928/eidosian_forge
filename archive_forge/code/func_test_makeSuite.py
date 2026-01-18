import functools
import sys
import types
import warnings
import unittest
def test_makeSuite(self):
    with self.assertWarns(DeprecationWarning) as w:
        suite = unittest.makeSuite(self.MyTestCase, prefix='check', sortUsing=self.reverse_three_way_cmp, suiteClass=self.MyTestSuite)
    self.assertEqual(w.filename, __file__)
    self.assertIsInstance(suite, self.MyTestSuite)
    expected = self.MyTestSuite([self.MyTestCase('check_2'), self.MyTestCase('check_1')])
    self.assertEqual(suite, expected)