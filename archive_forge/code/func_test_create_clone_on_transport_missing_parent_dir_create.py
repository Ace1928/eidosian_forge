from breezy import branch, errors, tests
from breezy.bzr import remote
from breezy.tests import per_branch
from breezy.transport import FileExists, NoSuchFile
def test_create_clone_on_transport_missing_parent_dir_create(self):
    tree = self.make_branch_and_tree('source')
    tree.commit('a commit')
    source = tree.branch
    target_transport = self.get_transport('subdir').clone('target')
    result = tree.branch.create_clone_on_transport(target_transport, create_prefix=True)
    self.assertEqual(source.last_revision(), result.last_revision())
    self.assertEqual(target_transport.base, result.controldir.root_transport.base)