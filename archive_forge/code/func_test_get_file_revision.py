from breezy import revision
from breezy.tests import TestCaseWithTransport
from breezy.tree import FileTimestampUnavailable
def test_get_file_revision(self):
    self.build_tree_contents([('a', b'initial')])
    self.t.add(['a'])
    revid1 = self.t.commit('add a')
    revid2 = self.t.commit('another change', allow_pointless=True)
    tree = self.t.branch.repository.revision_tree(revid2)
    self.assertEqual(revid1, tree.get_file_revision('a'))