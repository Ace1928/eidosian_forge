import operator
import sys
import types
import unittest
import abc
import pytest
import six
def test_assertRegex():

    class TestAssertRegex(unittest.TestCase):

        def test(self):
            with self.assertRaises(AssertionError):
                six.assertRegex(self, 'test', '^a')
            six.assertRegex(self, 'test', '^t')
    TestAssertRegex('test').test()