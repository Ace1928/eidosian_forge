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
def test_multiple_servers(self):
    transport1 = oslo_messaging.get_rpc_transport(self.conf, url='fake:')
    if self.exchange1 != self.exchange2:
        transport2 = oslo_messaging.get_rpc_transport(self.conf, url='fake:')
    else:
        transport2 = transport1

    class TestEndpoint(object):

        def __init__(self):
            self.pings = []

        def ping(self, ctxt, arg):
            self.pings.append(arg)

        def alive(self, ctxt):
            return 'alive'
    if self.multi_endpoints:
        endpoint1, endpoint2 = (TestEndpoint(), TestEndpoint())
    else:
        endpoint1 = endpoint2 = TestEndpoint()
    server1 = self._setup_server(transport1, endpoint1, topic=self.topic1, exchange=self.exchange1, server=self.server1)
    server2 = self._setup_server(transport2, endpoint2, topic=self.topic2, exchange=self.exchange2, server=self.server2)
    client1 = self._setup_client(transport1, topic=self.topic1, exchange=self.exchange1)
    client2 = self._setup_client(transport2, topic=self.topic2, exchange=self.exchange2)
    client1 = client1.prepare(server=self.server1)
    client2 = client2.prepare(server=self.server2)
    if self.fanout1:
        client1.call({}, 'alive')
        client1 = client1.prepare(fanout=True)
    if self.fanout2:
        client2.call({}, 'alive')
        client2 = client2.prepare(fanout=True)
    (client1.call if self.call1 else client1.cast)({}, 'ping', arg='1')
    (client2.call if self.call2 else client2.cast)({}, 'ping', arg='2')
    self._stop_server(client1.prepare(fanout=None), server1, topic=self.topic1, exchange=self.exchange1)
    self._stop_server(client2.prepare(fanout=None), server2, topic=self.topic2, exchange=self.exchange2)

    def check(pings, expect):
        self.assertEqual(len(expect), len(pings))
        for a in expect:
            self.assertIn(a, pings)
    if self.expect_either:
        check(endpoint1.pings + endpoint2.pings, self.expect_either)
    else:
        check(endpoint1.pings, self.expect1)
        check(endpoint2.pings, self.expect2)