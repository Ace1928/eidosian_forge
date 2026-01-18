from openstack import exceptions
from openstack.tests.functional.baremetal import base
def test_fake_hardware_get(self):
    driver = self.conn.baremetal.get_driver('fake-hardware')
    self.assertEqual('fake-hardware', driver.name)
    for iface in ('boot', 'deploy', 'management', 'power'):
        self.assertIn('fake', getattr(driver, 'enabled_%s_interfaces' % iface))
        self.assertEqual('fake', getattr(driver, 'default_%s_interface' % iface))
    self.assertNotEqual([], driver.hosts)