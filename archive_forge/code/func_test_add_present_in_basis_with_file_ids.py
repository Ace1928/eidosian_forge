from breezy import errors, tests
from breezy.bzr import inventory
from breezy.tests.matchers import HasLayout, HasPathRelations
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_add_present_in_basis_with_file_ids(self):
    tree = self.make_branch_and_tree('.')
    if not tree.supports_setting_file_ids():
        self.skipTest('tree does not support setting file ids')
    self.build_tree(['foo'])
    tree.add(['foo'], ids=[b'foo-id'])
    tree.commit('add foo')
    tree.unversion(['foo'])
    tree.add(['foo'], ids=[b'foo-id'])
    self.assertEqual(b'foo-id', tree.path2id('foo'))