import os
import requests
import subprocess
import time
import uuid
import concurrent.futures
from oslo_config import cfg
from testtools import matchers
import oslo_messaging
from oslo_messaging.tests.functional import utils
def test_endpoint_version_namespace(self):
    target = oslo_messaging.Target(topic='topic_' + str(uuid.uuid4()), server='server_' + str(uuid.uuid4()), namespace='Name1', version='7.5')

    class _endpoint(object):

        def __init__(self, target):
            self.target = target()

        def test(self, ctxt, echo):
            return echo
    transport = self.useFixture(utils.RPCTransportFixture(self.conf, self.rpc_url))
    self.useFixture(utils.RpcServerFixture(self.conf, self.rpc_url, target, executor='threading', endpoint=_endpoint(target)))
    client1 = utils.ClientStub(transport.transport, target, cast=False, timeout=5)
    self.assertEqual('Hi there', client1.test(echo='Hi there'))
    target2 = target()
    target2.version = '7.6'
    client2 = utils.ClientStub(transport.transport, target2, cast=False, timeout=5)
    self.assertRaises(oslo_messaging.rpc.client.RemoteError, client2.test, echo='Expect failure')
    target3 = oslo_messaging.Target(topic=target.topic, server=target.server, version=target.version, namespace='Name2')
    client3 = utils.ClientStub(transport.transport, target3, cast=False, timeout=5)
    self.assertRaises(oslo_messaging.rpc.client.RemoteError, client3.test, echo='Expect failure')