import copy
from unittest import mock
from keystoneauth1 import exceptions as ksa_exceptions
from keystoneauth1 import session as ksa_session
from openstack.config import cloud_region
from openstack.config import defaults
from openstack import exceptions
from openstack.tests.unit.config import base
from openstack import version as openstack_version
def test_get_endpoint_from_catalog(self):
    dns_override = 'https://override.dns.example.com'
    self.cloud.config.config['dns_endpoint_override'] = dns_override
    self.assertEqual('https://compute.example.com/v2.1/', self.cloud.config.get_endpoint_from_catalog('compute'))
    self.assertEqual('https://internal.compute.example.com/v2.1/', self.cloud.config.get_endpoint_from_catalog('compute', interface='internal'))
    self.assertIsNone(self.cloud.config.get_endpoint_from_catalog('compute', region_name='unknown-region'))
    self.assertEqual('https://dns.example.com', self.cloud.config.get_endpoint_from_catalog('dns'))