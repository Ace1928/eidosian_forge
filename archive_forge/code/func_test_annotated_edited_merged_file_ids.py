from breezy import config, tests
from breezy.urlutils import joinpath
from ..test_bedding import override_whoami
def test_annotated_edited_merged_file_ids(self):
    self._create_merged_file()
    out, err = self.run_bzr(['annotate', 'file', '--show-ids'])
    self.assertEqual('current: | local\n    rev1 | foo\nrev1.1.1 | bar\n    rev2 | baz\n    rev1 | gam\n', out)