from oslo_config import cfg
import testscenarios
from unittest import mock
import oslo_messaging
from oslo_messaging import exceptions
from oslo_messaging import serializer as msg_serializer
from oslo_messaging.tests import utils as test_utils
def test_call_serializer(self):
    self.config(rpc_response_timeout=None)
    transport = oslo_messaging.get_rpc_transport(self.conf, url='fake:')
    serializer = msg_serializer.NoOpSerializer()
    client = oslo_messaging.get_rpc_client(transport, oslo_messaging.Target(), serializer=serializer)
    transport._send = mock.Mock()
    kwargs = dict(wait_for_reply=True, timeout=None) if self.call else {}
    kwargs['retry'] = None
    if self.call:
        kwargs['call_monitor_timeout'] = None
    transport._send.return_value = self.retval
    serializer.serialize_entity = mock.Mock()
    serializer.deserialize_entity = mock.Mock()
    serializer.serialize_context = mock.Mock()

    def _stub(ctxt, arg):
        return 's' + arg
    msg = dict(method='foo', args=dict())
    for k, v in self.args.items():
        msg['args'][k] = 's' + v
    serializer.serialize_entity.side_effect = _stub
    if self.call:
        serializer.deserialize_entity.return_value = 'd' + self.retval
    serializer.serialize_context.return_value = dict(user='alice')
    method = client.call if self.call else client.cast
    retval = method(self.ctxt, 'foo', **self.args)
    if self.retval is not None:
        self.assertEqual('d' + self.retval, retval)
    transport._send.assert_called_once_with(oslo_messaging.Target(), dict(user='alice'), msg, transport_options=None, **kwargs)
    expected_calls = [mock.call(self.ctxt, arg) for arg in self.args]
    self.assertEqual(expected_calls, serializer.serialize_entity.mock_calls)
    if self.call:
        serializer.deserialize_entity.assert_called_once_with(self.ctxt, self.retval)
    serializer.serialize_context.assert_called_once_with(self.ctxt)