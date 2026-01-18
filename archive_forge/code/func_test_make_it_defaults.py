from openstack.compute.v2 import keypair
from openstack.tests.unit import base
def test_make_it_defaults(self):
    EXAMPLE_DEFAULT = EXAMPLE.copy()
    EXAMPLE_DEFAULT.pop('type')
    sot = keypair.Keypair(**EXAMPLE_DEFAULT)
    self.assertEqual(EXAMPLE['type'], sot.type)