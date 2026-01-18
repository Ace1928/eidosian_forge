from unittest import mock
from keystoneauth1 import adapter
from openstack.compute.v2 import flavor
from openstack.tests.unit import base
def test_make_basic(self):
    sot = flavor.Flavor(**BASIC_EXAMPLE)
    self.assertEqual(BASIC_EXAMPLE['id'], sot.id)
    self.assertEqual(BASIC_EXAMPLE['name'], sot.name)
    self.assertEqual(BASIC_EXAMPLE['description'], sot.description)
    self.assertEqual(BASIC_EXAMPLE['disk'], sot.disk)
    self.assertEqual(BASIC_EXAMPLE['os-flavor-access:is_public'], sot.is_public)
    self.assertEqual(BASIC_EXAMPLE['ram'], sot.ram)
    self.assertEqual(BASIC_EXAMPLE['vcpus'], sot.vcpus)
    self.assertEqual(BASIC_EXAMPLE['swap'], sot.swap)
    self.assertEqual(BASIC_EXAMPLE['OS-FLV-EXT-DATA:ephemeral'], sot.ephemeral)
    self.assertEqual(BASIC_EXAMPLE['OS-FLV-DISABLED:disabled'], sot.is_disabled)
    self.assertEqual(BASIC_EXAMPLE['rxtx_factor'], sot.rxtx_factor)