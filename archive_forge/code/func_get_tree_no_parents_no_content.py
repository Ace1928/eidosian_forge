import contextlib
from breezy import errors, tests, transform, transport
from breezy.bzr.workingtree_4 import (DirStateRevisionTree, WorkingTreeFormat4,
from breezy.git.tree import GitRevisionTree
from breezy.git.workingtree import GitWorkingTreeFormat
from breezy.revisiontree import RevisionTree
from breezy.tests import features
from breezy.tests.per_controldir.test_controldir import TestCaseWithControlDir
from breezy.tests.per_workingtree import make_scenario as wt_make_scenario
from breezy.tests.per_workingtree import make_scenarios as wt_make_scenarios
from breezy.workingtree import format_registry
def get_tree_no_parents_no_content(self, empty_tree, converter=None):
    """Make a tree with no parents and no contents from empty_tree.

        :param empty_tree: A working tree with no content and no parents to
            modify.
        """
    if empty_tree.supports_setting_file_ids():
        empty_tree.set_root_id(b'empty-root-id')
    return self._convert_tree(empty_tree, converter)