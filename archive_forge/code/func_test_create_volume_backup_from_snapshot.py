from openstack.block_storage.v3 import backup
from openstack.tests.unit import base
def test_create_volume_backup_from_snapshot(self):
    volume_id = '1234'
    backup_name = 'bak1'
    snapshot_id = '5678'
    bak1 = {'id': '5678', 'volume_id': volume_id, 'status': 'available', 'name': 'bak1'}
    self.register_uris([dict(method='POST', uri=self.get_mock_url('volumev3', 'public', append=['backups']), json={'backup': bak1}, validate=dict(json={'backup': {'name': backup_name, 'volume_id': volume_id, 'description': None, 'force': False, 'snapshot_id': snapshot_id, 'incremental': False}}))])
    self.cloud.create_volume_backup(volume_id, name=backup_name, snapshot_id=snapshot_id)
    self.assert_calls()