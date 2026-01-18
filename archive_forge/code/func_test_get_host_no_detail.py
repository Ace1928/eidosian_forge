from openstack.cloud import inventory
from openstack.tests.functional import base
def test_get_host_no_detail(self):
    host = self.inventory.get_host(self.server_id, expand=False)
    self.assertIsNotNone(host)
    self.assertEqual(host['name'], self.server_name)
    self.assertEqual(host['image']['id'], self.image.id)
    self.assertNotIn('links', host['image'])
    self.assertNotIn('name', host['name'])
    self.assertIn('ram', host['flavor'])
    host_found = False
    for host in self.inventory.list_hosts(expand=False):
        if host['id'] == self.server_id:
            host_found = True
            self._test_host_content(host)
    self.assertTrue(host_found)