from breezy import config, tests
from breezy.urlutils import joinpath
from ..test_bedding import override_whoami
def test_annotate_cmd_revision3(self):
    out, err = self.run_bzr('annotate hello.txt -r3')
    self.assertEqual('', err)
    self.assertEqualDiff('1   test@us | my helicopter\n3   user@te | your helicopter\n', out)