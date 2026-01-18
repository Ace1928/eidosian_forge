from io import BytesIO
from ... import errors, tests, ui
from . import TestCaseWithBranch
def test__get_check_refs(self):
    tree = self.make_branch_and_tree('.')
    revid = tree.commit('foo')
    self.assertEqual({('revision-existence', revid), ('lefthand-distance', revid)}, set(tree.branch._get_check_refs()))