from dulwich.tests import SkipTest, TestCase
from dulwich.tests.compat import utils
def test_git_version_4(self):
    self._version_str = b'git version 1.7.0.2'
    self.assertEqual((1, 7, 0, 2), utils.git_version())