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
def test_mandatory_call(self):
    if not self.rpc_url.startswith('rabbit://'):
        self.skipTest('backend does not support call monitoring')
    transport = self.useFixture(utils.RPCTransportFixture(self.conf, self.rpc_url))
    target = oslo_messaging.Target(topic='topic_' + str(uuid.uuid4()), server='server_' + str(uuid.uuid4()))
    options = oslo_messaging.TransportOptions(at_least_once=False)
    client1 = utils.ClientStub(transport.transport, target, cast=False, timeout=1, transport_options=options)
    self.assertRaises(oslo_messaging.MessagingTimeout, client1.delay)
    options2 = oslo_messaging.TransportOptions(at_least_once=True)
    client2 = utils.ClientStub(transport.transport, target, cast=False, timeout=60, transport_options=options2)
    self.assertRaises(oslo_messaging.MessageUndeliverable, client2.delay)