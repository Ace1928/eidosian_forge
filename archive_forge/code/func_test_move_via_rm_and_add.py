import os
from breezy import errors, osutils, tests
from breezy.tests import features
from breezy.tests.matchers import HasLayout, HasPathRelations
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_move_via_rm_and_add(self):
    """Move by remove and add-with-id"""
    self.build_tree(['a1', 'b1'])
    tree = self.make_branch_and_tree('.')
    if tree.supports_setting_file_ids():
        tree.add(['a1'], ids=[b'a1-id'])
    else:
        tree.add(['a1'])
    tree.commit('initial commit')
    tree.remove('a1', force=True, keep_files=False)
    if tree.supports_setting_file_ids():
        tree.add(['b1'], ids=[b'a1-id'])
    else:
        tree.add(['b1'])
    tree._validate()