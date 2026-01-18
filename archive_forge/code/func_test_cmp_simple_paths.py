import bisect
import os
import time
from ... import osutils, tests
from ...tests import features
from ...tests.scenarios import load_tests_apply_scenarios, multiply_scenarios
from ...tests.test_osutils import dir_reader_scenarios
from .. import _dirstate_helpers_py, dirstate
from . import test_dirstate
def test_cmp_simple_paths(self):
    """Compare against the empty string."""
    self.assertLtPathByDirblock([b'', b'a', b'ab', b'abc', b'a/b/c', b'b/d/e'])
    self.assertLtPathByDirblock([b'kl', b'ab/cd', b'ab/ef', b'gh/ij'])