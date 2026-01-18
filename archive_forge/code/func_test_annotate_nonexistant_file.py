from breezy import config, tests
from breezy.urlutils import joinpath
from ..test_bedding import override_whoami
def test_annotate_nonexistant_file(self):
    tree = self.make_branch_and_tree('.')
    self.build_tree(['file'])
    tree.add(['file'])
    tree.commit('add a file')
    out, err = self.run_bzr(['annotate', 'doesnotexist'], retcode=3)
    self.assertEqual('', out)
    self.assertEqual('brz: ERROR: doesnotexist is not versioned.\n', err)