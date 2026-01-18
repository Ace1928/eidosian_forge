from openstack import exceptions
from openstack.tests.functional.baremetal import base
def test_fake_hardware_list(self):
    drivers = self.conn.baremetal.drivers()
    self.assertIn('fake-hardware', [d.name for d in drivers])