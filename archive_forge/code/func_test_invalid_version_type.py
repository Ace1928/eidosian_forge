from oslo_config import cfg
import testscenarios
from unittest import mock
import oslo_messaging
from oslo_messaging import exceptions
from oslo_messaging import serializer as msg_serializer
from oslo_messaging.tests import utils as test_utils
def test_invalid_version_type(self):
    target = oslo_messaging.Target(topic='sometopic')
    transport = oslo_messaging.get_rpc_transport(self.conf, url='fake:')
    client = oslo_messaging.get_rpc_client(transport, target)
    self.assertRaises(exceptions.MessagingException, client.prepare, version='5')
    self.assertRaises(exceptions.MessagingException, client.prepare, version='5.a')
    self.assertRaises(exceptions.MessagingException, client.prepare, version='5.5.a')