from unittest import mock
from openstack.cloud import meta
from openstack.compute.v2 import server as _server
from openstack import connection
from openstack.tests import fakes
from openstack.tests.unit import base
def test_find_nova_addresses_key_name(self):
    addrs = {'public': [{'addr': '198.51.100.1', 'version': 4}], 'private': [{'addr': '192.0.2.5', 'version': 4}]}
    self.assertEqual(['198.51.100.1'], meta.find_nova_addresses(addrs, key_name='public'))
    self.assertEqual([], meta.find_nova_addresses(addrs, key_name='foo'))