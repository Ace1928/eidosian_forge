from oslo_config import cfg
import testscenarios
from unittest import mock
import oslo_messaging
from oslo_messaging import exceptions
from oslo_messaging import serializer as msg_serializer
from oslo_messaging.tests import utils as test_utils
def test_cast_call_with_transport_options(self):
    self.config(rpc_response_timeout=None)
    transport = oslo_messaging.get_rpc_transport(self.conf, url='fake:')
    transport_options = oslo_messaging.TransportOptions(at_least_once=True)
    client = oslo_messaging.get_rpc_client(transport, oslo_messaging.Target(), transport_options=transport_options)
    transport._send = mock.Mock()
    msg = dict(method='foo', args=self.args)
    kwargs = {'retry': None, 'transport_options': transport_options}
    if self.call:
        kwargs['wait_for_reply'] = True
        kwargs['timeout'] = None
        kwargs['call_monitor_timeout'] = None
    method = client.call if self.call else client.cast
    method(self.ctxt, 'foo', **self.args)
    self.assertTrue(transport_options.at_least_once)
    transport._send.assert_called_once_with(oslo_messaging.Target(), self.ctxt, msg, **kwargs)