import bisect
import os
import time
from ... import osutils, tests
from ...tests import features
from ...tests.scenarios import load_tests_apply_scenarios, multiply_scenarios
from ...tests.test_osutils import dir_reader_scenarios
from .. import _dirstate_helpers_py, dirstate
from . import test_dirstate
def assertBisect(self, dirblocks, split_dirblocks, path, *args, **kwargs):
    """Assert that bisect_split works like bisect_left on the split paths.

        :param dirblocks: A list of (path, [info]) pairs.
        :param split_dirblocks: A list of ((split, path), [info]) pairs.
        :param path: The path we are indexing.

        All other arguments will be passed along.
        """
    bisect_dirblock = self.get_bisect_dirblock()
    self.assertIsInstance(dirblocks, list)
    bisect_split_idx = bisect_dirblock(dirblocks, path, *args, **kwargs)
    split_dirblock = (path.split(b'/'), [])
    bisect_left_idx = bisect.bisect_left(split_dirblocks, split_dirblock, *args)
    self.assertEqual(bisect_left_idx, bisect_split_idx, 'bisect_split disagreed. %s != %s for key %r' % (bisect_left_idx, bisect_split_idx, path))