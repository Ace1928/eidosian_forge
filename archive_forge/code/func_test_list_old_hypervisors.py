from unittest import mock
import uuid
import testtools
from openstack.config import cloud_region
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_list_old_hypervisors(self):
    """This test verifies that calling list_hypervisors on a pre-2.53 cloud
        calls the old version."""
    self.use_compute_discovery(compute_version_json='old-compute-version.json')
    self.register_uris([dict(method='GET', uri='https://compute.example.com/v2.1/os-hypervisors/detail', json={'hypervisors': [fakes.make_fake_hypervisor('1', 'testserver1'), fakes.make_fake_hypervisor('2', 'testserver2')]})])
    r = self.cloud.list_hypervisors()
    self.assertEqual(2, len(r))
    self.assertEqual('testserver1', r[0]['name'])
    self.assertEqual('testserver2', r[1]['name'])
    self.assert_calls()