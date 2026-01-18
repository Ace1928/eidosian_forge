from breezy import branch, errors, tests
from breezy.bzr import remote
from breezy.tests import per_branch
from breezy.transport import FileExists, NoSuchFile
def test_create_clone_on_transport_missing_parent_dir(self):
    tree = self.make_branch_and_tree('source')
    tree.commit('a commit')
    source = tree.branch
    target_transport = self.get_transport('subdir').clone('target')
    self.assertRaises(NoSuchFile, tree.branch.create_clone_on_transport, target_transport)
    self.assertFalse(self.get_transport('.').has('subdir'))