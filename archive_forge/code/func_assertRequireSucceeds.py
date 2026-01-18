from dulwich.tests import SkipTest, TestCase
from dulwich.tests.compat import utils
def assertRequireSucceeds(self, required_version):
    try:
        utils.require_git_version(required_version)
    except SkipTest:
        self.fail()