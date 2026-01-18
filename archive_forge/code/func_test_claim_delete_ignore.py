from unittest import mock
from openstack.message.v2 import _proxy
from openstack.message.v2 import claim
from openstack.message.v2 import message
from openstack.message.v2 import queue
from openstack.message.v2 import subscription
from openstack import proxy as proxy_base
from openstack.tests.unit import test_proxy_base
def test_claim_delete_ignore(self):
    self.verify_delete(self.proxy.delete_claim, claim.Claim, ignore_missing=True, method_args=['test_queue', 'test_claim'], expected_args=['test_claim'], expected_kwargs={'queue_name': 'test_queue', 'ignore_missing': True})