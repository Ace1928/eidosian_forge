import bisect
import os
import time
from ... import osutils, tests
from ...tests import features
from ...tests.scenarios import load_tests_apply_scenarios, multiply_scenarios
from ...tests.test_osutils import dir_reader_scenarios
from .. import _dirstate_helpers_py, dirstate
from . import test_dirstate
def test_cmp_empty(self):
    """Compare against the empty string."""
    self.assertCmpByDirs(0, b'', b'')
    self.assertCmpByDirs(1, b'a', b'')
    self.assertCmpByDirs(1, b'ab', b'')
    self.assertCmpByDirs(1, b'abc', b'')
    self.assertCmpByDirs(1, b'abcd', b'')
    self.assertCmpByDirs(1, b'abcde', b'')
    self.assertCmpByDirs(1, b'abcdef', b'')
    self.assertCmpByDirs(1, b'abcdefg', b'')
    self.assertCmpByDirs(1, b'abcdefgh', b'')
    self.assertCmpByDirs(1, b'abcdefghi', b'')
    self.assertCmpByDirs(1, b'test/ing/a/path/', b'')