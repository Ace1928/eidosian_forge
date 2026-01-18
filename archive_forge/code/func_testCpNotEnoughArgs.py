from gslib.utils import system_util
from gslib.utils.user_agent_helper import GetUserAgent
import gslib.tests.testcase as testcase
import six
from six import add_move, MovedModule
from six.moves import mock
@mock.patch.object(system_util, 'InvokedViaCloudSdk')
def testCpNotEnoughArgs(self, mock_invoked):
    mock_invoked.return_value = False
    self.assertRegex(GetUserAgent(['cp']), 'command/cp$')
    self.assertRegex(GetUserAgent(['cp', '1.txt']), 'command/cp$')
    self.assertRegex(GetUserAgent(['cp', '-r', '1.ts']), 'command/cp$')