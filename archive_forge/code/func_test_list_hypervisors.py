from unittest import mock
import uuid
import testtools
from openstack.config import cloud_region
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_list_hypervisors(self):
    """This test verifies that calling list_hypervisors results in a call
        to nova client."""
    uuid1 = uuid.uuid4().hex
    uuid2 = uuid.uuid4().hex
    self.use_compute_discovery()
    self.register_uris([dict(method='GET', uri='https://compute.example.com/v2.1/os-hypervisors/detail', json={'hypervisors': [fakes.make_fake_hypervisor(uuid1, 'testserver1'), fakes.make_fake_hypervisor(uuid2, 'testserver2')]}, validate={'headers': {'OpenStack-API-Version': 'compute 2.53'}})])
    r = self.cloud.list_hypervisors()
    self.assertEqual(2, len(r))
    self.assertEqual('testserver1', r[0]['name'])
    self.assertEqual(uuid1, r[0]['id'])
    self.assertEqual('testserver2', r[1]['name'])
    self.assertEqual(uuid2, r[1]['id'])
    self.assert_calls()