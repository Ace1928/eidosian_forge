import operator
import sys
import types
import unittest
import abc
import pytest
import six
def test_assertRaisesRegex():

    class TestAssertRaisesRegex(unittest.TestCase):

        def test(self):
            with six.assertRaisesRegex(self, AssertionError, '^Foo'):
                raise AssertionError('Foo')
            with self.assertRaises(AssertionError):
                with six.assertRaisesRegex(self, AssertionError, '^Foo'):
                    raise AssertionError('Bar')
    TestAssertRaisesRegex('test').test()