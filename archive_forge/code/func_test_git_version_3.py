from dulwich.tests import SkipTest, TestCase
from dulwich.tests.compat import utils
def test_git_version_3(self):
    self._version_str = b'git version 1.6.6'
    self.assertEqual((1, 6, 6, 0), utils.git_version())