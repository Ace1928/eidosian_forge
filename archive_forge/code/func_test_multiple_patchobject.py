import os
import sys
import six
import unittest2 as unittest
from mock.tests import support
from mock.tests.support import SomeClass, is_instance, callable
from mock import (
from mock.mock import _patch, _get_target
def test_multiple_patchobject(self):

    class Something(object):
        attribute = sentinel.Original
        next_attribute = sentinel.Original2

    @patch.object(Something, 'attribute', sentinel.Patched)
    @patch.object(Something, 'next_attribute', sentinel.Patched2)
    def test():
        self.assertEqual(Something.attribute, sentinel.Patched, 'unpatched')
        self.assertEqual(Something.next_attribute, sentinel.Patched2, 'unpatched')
    test()
    self.assertEqual(Something.attribute, sentinel.Original, 'patch not restored')
    self.assertEqual(Something.next_attribute, sentinel.Original2, 'patch not restored')