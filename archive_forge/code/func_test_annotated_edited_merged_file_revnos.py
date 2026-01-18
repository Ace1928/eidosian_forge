from breezy import config, tests
from breezy.urlutils import joinpath
from ..test_bedding import override_whoami
def test_annotated_edited_merged_file_revnos(self):
    wt = self._create_merged_file()
    out, err = self.run_bzr(['annotate', 'file'])
    email = config.extract_email_address(wt.branch.get_config_stack().get('email'))
    self.assertEqual('3?    %-7s | local\n1     test@ho | foo\n1.1.1 test@ho | bar\n2     test@ho | baz\n1     test@ho | gam\n' % email[:7], out)