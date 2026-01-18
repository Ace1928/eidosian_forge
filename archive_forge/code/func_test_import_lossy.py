from breezy.revision import NULL_REVISION
from breezy.tests import TestCaseWithTransport
def test_import_lossy(self):
    tree1 = self.make_branch_and_tree('branch1')
    tree1.commit('1st post')
    revid = tree1.commit('2st post', allow_pointless=True)
    branch2 = self.make_branch('branch2')
    ret = branch2.import_last_revision_info_and_tags(tree1.branch, 2, revid, lossy=True)
    self.assertIsInstance(ret, tuple)
    self.assertIsInstance(ret[0], int)
    self.assertIsInstance(ret[1], bytes)