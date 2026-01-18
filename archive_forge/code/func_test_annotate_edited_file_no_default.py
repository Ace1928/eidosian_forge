from breezy import config, tests
from breezy.urlutils import joinpath
from ..test_bedding import override_whoami
def test_annotate_edited_file_no_default(self):
    override_whoami(self)
    tree = self._setup_edited_file()
    out, err = self.run_bzr('annotate file')
    self.assertEqual('1   test@ho | foo\n2?  local u | bar\n1   test@ho | gam\n', out)