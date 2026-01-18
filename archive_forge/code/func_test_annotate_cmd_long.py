from breezy import config, tests
from breezy.urlutils import joinpath
from ..test_bedding import override_whoami
def test_annotate_cmd_long(self):
    out, err = self.run_bzr('annotate hello.txt --long')
    self.assertEqual('', err)
    self.assertEqualDiff('1   test@user 20061212 | my helicopter\n3   user@test 20061213 | your helicopter\n4   user@test 20061213 | all of\n                       | our helicopters\n', out)