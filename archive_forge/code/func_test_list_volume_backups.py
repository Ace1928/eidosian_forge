from openstack.block_storage.v3 import backup
from openstack.tests.unit import base
def test_list_volume_backups(self):
    backup = {'id': '6ff16bdf-44d5-4bf9-b0f3-687549c76414', 'status': 'available'}
    search_opts = {'status': 'available'}
    self.register_uris([dict(method='GET', uri=self.get_mock_url('volumev3', 'public', append=['backups', 'detail'], qs_elements=['='.join(i) for i in search_opts.items()]), json={'backups': [backup]})])
    result = self.cloud.list_volume_backups(True, search_opts)
    self.assertEqual(len(result), 1)
    self._compare_backups(backup, result[0])
    self.assert_calls()