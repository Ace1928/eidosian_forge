from unittest import mock
from keystoneauth1 import adapter
from openstack.compute.v2 import flavor
from openstack.tests.unit import base
def test_make_basic_swap(self):
    sot = flavor.Flavor(id=IDENTIFIER, swap='')
    self.assertEqual(0, sot.swap)
    sot1 = flavor.Flavor(id=IDENTIFIER, swap=0)
    self.assertEqual(0, sot1.swap)