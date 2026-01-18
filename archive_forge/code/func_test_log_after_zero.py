import threading
from unittest import mock
import eventlet
import fixtures
from oslo_config import cfg
from oslo_utils import eventletutils
import testscenarios
import oslo_messaging
from oslo_messaging import rpc
from oslo_messaging.rpc import dispatcher
from oslo_messaging.rpc import server as rpc_server_module
from oslo_messaging import server as server_module
from oslo_messaging.tests import utils as test_utils
@mock.patch.object(server_module, 'LOG')
def test_log_after_zero(self, mock_log):
    self.assertRaises(server_module.TaskTimeout, self.server.stop, log_after=0, timeout=2)
    self.assertFalse(mock_log.warning.called)