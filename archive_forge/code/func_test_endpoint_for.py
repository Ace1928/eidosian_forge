from unittest import mock
import uuid
import testtools
from openstack import connection
from openstack import exceptions
from openstack.tests.unit import base
from openstack import utils
def test_endpoint_for(self):
    dns_override = 'https://override.dns.example.com'
    self.cloud.config.config['dns_endpoint_override'] = dns_override
    self.assertEqual('https://compute.example.com/v2.1/', self.cloud.endpoint_for('compute'))
    self.assertEqual('https://internal.compute.example.com/v2.1/', self.cloud.endpoint_for('compute', interface='internal'))
    self.assertIsNone(self.cloud.endpoint_for('compute', region_name='unknown-region'))
    self.assertEqual(dns_override, self.cloud.endpoint_for('dns'))