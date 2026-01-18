from oslo_config import cfg
import testscenarios
from unittest import mock
import oslo_messaging
from oslo_messaging import exceptions
from oslo_messaging import serializer as msg_serializer
from oslo_messaging.tests import utils as test_utils
def test_call_fanout(self):
    transport = oslo_messaging.get_rpc_transport(self.conf, url='fake:')
    client = oslo_messaging.get_rpc_client(transport, oslo_messaging.Target(**self.target))
    if self.prepare is not _notset:
        client = client.prepare(**self.prepare)
    self.assertRaises(exceptions.InvalidTarget, client.call, {}, 'foo')