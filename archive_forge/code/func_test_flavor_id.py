from unittest import mock
from keystoneauth1 import adapter
from openstack.compute.v2 import flavor
from openstack.tests.unit import base
def test_flavor_id(self):
    id = 'fake_id'
    sot = flavor.Flavor(id=id)
    self.assertEqual(sot.id, id)
    sot = flavor.Flavor(name=id)
    self.assertEqual(sot.id, id)
    self.assertEqual(sot.name, id)
    sot = flavor.Flavor(original_name=id)
    self.assertEqual(sot.id, id)
    self.assertEqual(sot.original_name, id)