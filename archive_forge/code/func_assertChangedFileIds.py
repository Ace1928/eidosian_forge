import bisect
import os
import time
from ... import osutils, tests
from ...tests import features
from ...tests.scenarios import load_tests_apply_scenarios, multiply_scenarios
from ...tests.test_osutils import dir_reader_scenarios
from .. import _dirstate_helpers_py, dirstate
from . import test_dirstate
def assertChangedFileIds(self, expected, tree):
    with tree.lock_read():
        file_ids = [info.file_id for info in tree.iter_changes(tree.basis_tree())]
    self.assertEqual(sorted(expected), sorted(file_ids))