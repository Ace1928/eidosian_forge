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
def test_client_call_timeout(self):
    transport = oslo_messaging.get_rpc_transport(self.conf, url='fake:')
    finished = False
    wait = threading.Condition()

    class TestEndpoint(object):

        def ping(self, ctxt, arg):
            with wait:
                if not finished:
                    wait.wait()
    server_thread = self._setup_server(transport, TestEndpoint())
    client = self._setup_client(transport)
    try:
        client.prepare(timeout=0).call({}, 'ping', arg='foo')
    except Exception as ex:
        self.assertIsInstance(ex, oslo_messaging.MessagingTimeout, ex)
    else:
        self.assertTrue(False)
    with wait:
        finished = True
        wait.notify()
    self._stop_server(client, server_thread)