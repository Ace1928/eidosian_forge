from breezy import config, tests
from breezy.urlutils import joinpath
from ..test_bedding import override_whoami
def test_no_mail(self):
    out, err = self.run_bzr('annotate nomail.txt')
    self.assertEqual('', err)
    self.assertEqualDiff('2   no mail | nomail\n', out)