from oslo_serialization import jsonutils
from novaclient.tests.functional import base
def test_extended_server_attributes(self):
    server, volume = self._create_server_and_attach_volume()
    table = self.nova('show %s' % server.id)
    for attr in ['OS-EXT-SRV-ATTR:host', 'OS-EXT-SRV-ATTR:hypervisor_hostname', 'OS-EXT-SRV-ATTR:instance_name']:
        self._get_value_from_the_table(table, attr)
    volume_attr = self._get_value_from_the_table(table, 'os-extended-volumes:volumes_attached')
    self.assertIn('id', jsonutils.loads(volume_attr)[0])