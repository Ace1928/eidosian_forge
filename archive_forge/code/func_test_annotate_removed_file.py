from breezy import config, tests
from breezy.urlutils import joinpath
from ..test_bedding import override_whoami
def test_annotate_removed_file(self):
    tree = self.make_branch_and_tree('.')
    self.build_tree_contents([('empty', b'')])
    tree.add('empty')
    tree.commit('add empty file')
    tree.remove('empty')
    tree.commit('remove empty file')
    out, err = self.run_bzr(['annotate', '-r1', 'empty'])
    self.assertEqual('', out)