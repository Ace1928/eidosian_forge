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
def test_logging_explicit_wait(self, mock_log):
    log_event = eventletutils.Event()
    mock_log.warning.side_effect = lambda _, __: log_event.set()
    thread = eventlet.spawn(self.server.stop, log_after=1)
    log_event.wait()
    self.assertTrue(mock_log.warning.called)
    thread.kill()