import bisect
import os
import time
from ... import osutils, tests
from ...tests import features
from ...tests.scenarios import load_tests_apply_scenarios, multiply_scenarios
from ...tests.test_osutils import dir_reader_scenarios
from .. import _dirstate_helpers_py, dirstate
from . import test_dirstate
def split_for_dirblocks(self, paths):
    dir_split_paths = []
    for path in paths:
        dirname, basename = os.path.split(path)
        dir_split_paths.append((dirname.split(b'/'), basename))
    dir_split_paths.sort()
    return dir_split_paths