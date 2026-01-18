import bisect
import os
import time
from ... import osutils, tests
from ...tests import features
from ...tests.scenarios import load_tests_apply_scenarios, multiply_scenarios
from ...tests.test_osutils import dir_reader_scenarios
from .. import _dirstate_helpers_py, dirstate
from . import test_dirstate
def test_exceptions_raised(self):
    tree = self.make_branch_and_tree('tree')
    self.build_tree(['tree/file', 'tree/dir/', 'tree/dir/sub', 'tree/dir2/', 'tree/dir2/sub2'])
    tree.add(['file', 'dir', 'dir/sub', 'dir2', 'dir2/sub2'])
    tree.commit('first commit')
    tree.lock_read()
    self.addCleanup(tree.unlock)
    basis_tree = tree.basis_tree()

    def is_inside_raises(*args, **kwargs):
        raise RuntimeError('stop this')
    self.overrideAttr(dirstate, 'is_inside', is_inside_raises)
    try:
        from .. import _dirstate_helpers_pyx
    except ImportError:
        pass
    else:
        self.overrideAttr(_dirstate_helpers_pyx, 'is_inside', is_inside_raises)
    self.overrideAttr(osutils, 'is_inside', is_inside_raises)
    self.assertListRaises(RuntimeError, tree.iter_changes, basis_tree)