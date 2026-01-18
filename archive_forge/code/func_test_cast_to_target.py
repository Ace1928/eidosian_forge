from oslo_config import cfg
import testscenarios
from unittest import mock
import oslo_messaging
from oslo_messaging import exceptions
from oslo_messaging import serializer as msg_serializer
from oslo_messaging.tests import utils as test_utils
def test_cast_to_target(self):
    target = oslo_messaging.Target(**self.ctor)
    expect_target = oslo_messaging.Target(**self.expect)
    transport = oslo_messaging.get_rpc_transport(self.conf, url='fake:')
    client = oslo_messaging.get_rpc_client(transport, target)
    transport._send = mock.Mock()
    msg = dict(method='foo', args={})
    if 'namespace' in self.expect:
        msg['namespace'] = self.expect['namespace']
    if 'version' in self.expect:
        msg['version'] = self.expect['version']
    if self.prepare:
        client = client.prepare(**self.prepare)
        if self.double_prepare:
            client = client.prepare(**self.prepare)
    client.cast({}, 'foo')
    transport._send.assert_called_once_with(expect_target, {}, msg, retry=None, transport_options=None)