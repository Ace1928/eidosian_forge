from gslib.utils import system_util
from gslib.utils.user_agent_helper import GetUserAgent
import gslib.tests.testcase as testcase
import six
from six import add_move, MovedModule
from six.moves import mock
@mock.patch('gslib.VERSION', '4_test')
def testNoArgs(self):
    self.assertRegex(GetUserAgent([]), '^ gsutil/4_test \\([^\\)]+\\)')