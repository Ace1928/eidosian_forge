import os
from breezy import errors, osutils, tests
from breezy.tests import features
from breezy.tests.matchers import HasLayout, HasPathRelations
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_move_correct_call_unnamed(self):
    """tree.move has the deprecated parameter 'to_name'.
        It has been replaced by 'to_dir' for consistency.
        Test the new API using unnamed parameter
        """
    self.build_tree(['a1', 'sub1/'])
    tree = self.make_branch_and_tree('.')
    tree.add(['a1', 'sub1'])
    tree.commit('initial commit')
    self.assertEqual([('a1', 'sub1/a1')], tree.move(['a1'], 'sub1', after=False))
    tree._validate()