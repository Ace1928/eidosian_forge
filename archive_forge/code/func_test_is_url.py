import os
import sys
from .. import osutils, urlutils
from ..errors import PathNotChild
from . import TestCase, TestCaseInTempDir, TestSkipped, features
def test_is_url(self):
    self.assertTrue(urlutils.is_url('http://foo/bar'))
    self.assertTrue(urlutils.is_url('bzr+ssh://foo/bar'))
    self.assertTrue(urlutils.is_url('lp:foo/bar'))
    self.assertTrue(urlutils.is_url('file:///foo/bar'))
    self.assertFalse(urlutils.is_url(''))
    self.assertFalse(urlutils.is_url('foo'))
    self.assertFalse(urlutils.is_url('foo/bar'))
    self.assertFalse(urlutils.is_url('/foo'))
    self.assertFalse(urlutils.is_url('/foo/bar'))
    self.assertFalse(urlutils.is_url('C:/'))
    self.assertFalse(urlutils.is_url('C:/foo'))
    self.assertFalse(urlutils.is_url('C:/foo/bar'))