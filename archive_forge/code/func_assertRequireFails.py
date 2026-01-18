from dulwich.tests import SkipTest, TestCase
from dulwich.tests.compat import utils
def assertRequireFails(self, required_version):
    self.assertRaises(SkipTest, utils.require_git_version, required_version)