from unittest import mock
from keystoneauth1 import adapter
from openstack.compute.v2 import flavor
from openstack.tests.unit import base
def test_make_defaults(self):
    sot = flavor.Flavor(**DEFAULTS_EXAMPLE)
    self.assertEqual(DEFAULTS_EXAMPLE['original_name'], sot.name)
    self.assertEqual(0, sot.disk)
    self.assertEqual(True, sot.is_public)
    self.assertEqual(0, sot.ram)
    self.assertEqual(0, sot.vcpus)
    self.assertEqual(0, sot.swap)
    self.assertEqual(0, sot.ephemeral)
    self.assertEqual(IDENTIFIER, sot.id)