from unittest import mock
from oslo_config import cfg
import oslo_messaging as messaging
from oslo_messaging import conffixture as messaging_conffixture
from oslo_messaging import exceptions as oslomsg_exc
import testtools
from neutron_lib import fixture
from neutron_lib import rpc
from neutron_lib.tests import _base as base
def test_method_timeout_config_ceiling(self):
    rpc.TRANSPORT.conf.rpc_response_timeout = 10
    for i in range(5):
        with testtools.ExpectedException(messaging.MessagingTimeout):
            self.client.call(self.call_context, 'method_1')
    self.assertEqual(rpc.TRANSPORT.conf.rpc_response_max_timeout, rpc._BackingOffContextWrapper._METHOD_TIMEOUTS['method_1'])
    with testtools.ExpectedException(messaging.MessagingTimeout):
        self.client.call(self.call_context, 'method_1')
    self.assertEqual(rpc.TRANSPORT.conf.rpc_response_max_timeout, rpc._BackingOffContextWrapper._METHOD_TIMEOUTS['method_1'])