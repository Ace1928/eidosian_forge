from breezy import config, tests
from breezy.urlutils import joinpath
from ..test_bedding import override_whoami
def test_annotate_edited_file_show_ids(self):
    tree = self._setup_edited_file()
    self.overrideEnv('BRZ_EMAIL', 'current@host2')
    out, err = self.run_bzr('annotate file --show-ids')
    self.assertEqual('    rev1 | foo\ncurrent: | bar\n    rev1 | gam\n', out)