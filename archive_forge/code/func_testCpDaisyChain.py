from gslib.utils import system_util
from gslib.utils.user_agent_helper import GetUserAgent
import gslib.tests.testcase as testcase
import six
from six import add_move, MovedModule
from six.moves import mock
def testCpDaisyChain(self):
    self.assertRegex(GetUserAgent(['cp', '-r', '-Z', 'gs://src', 's3://dst']), 'command/cp-DaisyChain')
    self.assertRegex(GetUserAgent(['mv', 'gs://src/1.txt', 's3://dst/1.txt']), 'command/mv-DaisyChain')
    self.assertRegex(GetUserAgent(['rsync', '-r', 'gs://src', 's3://dst']), 'command/rsync-DaisyChain')