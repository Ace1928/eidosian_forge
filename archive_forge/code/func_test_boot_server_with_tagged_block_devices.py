from tempest.lib import exceptions
from novaclient.tests.functional import base
def test_boot_server_with_tagged_block_devices(self):
    server_info = self.nova('boot', params='%(name)s --flavor %(flavor)s --poll --nic net-id=%(net-uuid)s --block-device source=image,dest=volume,id=%(image)s,size=1,bootindex=0,shutdown=remove,tag=bar' % {'name': self.name_generate(), 'flavor': self.flavor.id, 'net-uuid': self.network.id, 'image': self.image.id})
    server_id = self._get_value_from_the_table(server_info, 'id')
    self.client.servers.delete(server_id)
    self.wait_for_resource_delete(server_id, self.client.servers)