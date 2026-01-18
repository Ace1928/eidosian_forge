from breezy import config, tests
from breezy.urlutils import joinpath
from ..test_bedding import override_whoami
def test_annotate_cmd_revision(self):
    out, err = self.run_bzr('annotate hello.txt -r1')
    self.assertEqual('', err)
    self.assertEqualDiff('1   test@us | my helicopter\n', out)