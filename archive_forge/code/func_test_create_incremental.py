from unittest import mock
from keystoneauth1 import adapter
from openstack.block_storage.v2 import backup
from openstack import exceptions
from openstack.tests.unit import base
def test_create_incremental(self):
    sot = backup.Backup(is_incremental=True)
    sot2 = backup.Backup(is_incremental=False)
    create_response = mock.Mock()
    create_response.status_code = 200
    create_response.json.return_value = {}
    create_response.headers = {}
    self.sess.post.return_value = create_response
    sot.create(self.sess)
    self.sess.post.assert_called_with('/backups', headers={}, json={'backup': {'incremental': True}}, microversion=None, params={})
    sot2.create(self.sess)
    self.sess.post.assert_called_with('/backups', headers={}, json={'backup': {'incremental': False}}, microversion=None, params={})