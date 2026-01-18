import bisect
import os
import time
from ... import osutils, tests
from ...tests import features
from ...tests.scenarios import load_tests_apply_scenarios, multiply_scenarios
from ...tests.test_osutils import dir_reader_scenarios
from .. import _dirstate_helpers_py, dirstate
from . import test_dirstate
def test_involved(self):
    """This is where bisect_left diverges slightly."""
    paths = [b'', b'a', b'a/a', b'a/a/a', b'a/a/z', b'a/a-a', b'a/a-z', b'a/z', b'a/z/a', b'a/z/z', b'a/z-a', b'a/z-z', b'a-a', b'a-z', b'z', b'z/a/a', b'z/a/z', b'z/a-a', b'z/a-z', b'z/z', b'z/z/a', b'z/z/z', b'z/z-a', b'z/z-z', b'z-a', b'z-z']
    dirblocks, split_dirblocks = self.paths_to_dirblocks(paths)
    for path in paths:
        self.assertBisect(dirblocks, split_dirblocks, path)