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
def test_constructor_with_unrecognized_executor(self):
    transport = oslo_messaging.get_rpc_transport(self.conf, url='fake:')
    target = oslo_messaging.Target(topic='foo', server='bar')
    endpoints = [object()]
    serializer = object()
    access_policy = dispatcher.DefaultRPCAccessPolicy
    self.assertRaises(server_module.ExecutorLoadFailure, oslo_messaging.get_rpc_server, transport=transport, target=target, endpoints=endpoints, serializer=serializer, access_policy=access_policy, executor='boom')