from breezy import config, tests
from breezy.urlutils import joinpath
from ..test_bedding import override_whoami
def test_annotate_cmd_revspec_branch(self):
    tree = self._setup_edited_file('trunk')
    tree.branch.create_checkout(self.get_url('work'), lightweight=True)
    out, err = self.run_bzr(['annotate', 'file', '-r', 'branch:../trunk'], working_dir='work')
    self.assertEqual('', err)
    self.assertEqual('1   test@ho | foo\n            | gam\n', out)