import datetime
from fixtures import TimeoutException
from openstack import exceptions
from openstack.tests.functional import base
from openstack import utils
def test_attach_volume_create_snapshot(self):
    self.skipTest('Volume functional tests temporarily disabled')
    server_name = self.getUniqueString()
    self.addCleanup(self._cleanup_servers_and_volumes, server_name)
    server = self.user_cloud.create_server(name=server_name, image=self.image, flavor=self.flavor, wait=True)
    volume = self.user_cloud.create_volume(1)
    vol_attachment = self.user_cloud.attach_volume(server, volume)
    for key in ('device', 'serverId', 'volumeId'):
        self.assertIn(key, vol_attachment)
        self.assertTrue(vol_attachment[key])
    snapshot = self.user_cloud.create_volume_snapshot(volume_id=volume.id, force=True, wait=True)
    self.addCleanup(self.user_cloud.delete_volume_snapshot, snapshot['id'])
    self.assertIsNotNone(snapshot)