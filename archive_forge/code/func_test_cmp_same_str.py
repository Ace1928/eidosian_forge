import bisect
import os
import time
from ... import osutils, tests
from ...tests import features
from ...tests.scenarios import load_tests_apply_scenarios, multiply_scenarios
from ...tests.test_osutils import dir_reader_scenarios
from .. import _dirstate_helpers_py, dirstate
from . import test_dirstate
def test_cmp_same_str(self):
    """Compare the same string"""
    self.assertCmpByDirs(0, b'a', b'a')
    self.assertCmpByDirs(0, b'ab', b'ab')
    self.assertCmpByDirs(0, b'abc', b'abc')
    self.assertCmpByDirs(0, b'abcd', b'abcd')
    self.assertCmpByDirs(0, b'abcde', b'abcde')
    self.assertCmpByDirs(0, b'abcdef', b'abcdef')
    self.assertCmpByDirs(0, b'abcdefg', b'abcdefg')
    self.assertCmpByDirs(0, b'abcdefgh', b'abcdefgh')
    self.assertCmpByDirs(0, b'abcdefghi', b'abcdefghi')
    self.assertCmpByDirs(0, b'testing a long string', b'testing a long string')
    self.assertCmpByDirs(0, b'x' * 10000, b'x' * 10000)
    self.assertCmpByDirs(0, b'a/b', b'a/b')
    self.assertCmpByDirs(0, b'a/b/c', b'a/b/c')
    self.assertCmpByDirs(0, b'a/b/c/d', b'a/b/c/d')
    self.assertCmpByDirs(0, b'a/b/c/d/e', b'a/b/c/d/e')