from breezy import config, tests
from breezy.urlutils import joinpath
from ..test_bedding import override_whoami
def test_annotate_empty_file_show_ids(self):
    tree = self.make_branch_and_tree('.')
    self.build_tree_contents([('empty', b'')])
    tree.add('empty')
    tree.commit('add empty file')
    out, err = self.run_bzr(['annotate', '--show-ids', 'empty'])
    self.assertEqual('', out)