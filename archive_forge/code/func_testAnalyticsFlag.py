from gslib.utils import system_util
from gslib.utils.user_agent_helper import GetUserAgent
import gslib.tests.testcase as testcase
import six
from six import add_move, MovedModule
from six.moves import mock
def testAnalyticsFlag(self):
    self.assertRegex(GetUserAgent([], False), 'analytics/enabled')
    self.assertRegex(GetUserAgent([], True), 'analytics/disabled')