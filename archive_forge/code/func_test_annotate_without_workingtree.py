from breezy import config, tests
from breezy.urlutils import joinpath
from ..test_bedding import override_whoami
def test_annotate_without_workingtree(self):
    tree = self.make_branch_and_tree('.')
    self.build_tree_contents([('empty', b'')])
    tree.add('empty')
    tree.commit('add empty file')
    bzrdir = tree.branch.controldir
    bzrdir.destroy_workingtree()
    self.assertFalse(bzrdir.has_workingtree())
    out, err = self.run_bzr(['annotate', 'empty'])
    self.assertEqual('', out)