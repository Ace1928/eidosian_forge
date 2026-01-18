from unittest import mock
from keystoneauth1 import adapter
from openstack.block_storage.v3 import transfer
from openstack import resource
from openstack.tests.unit import base
@mock.patch('openstack.utils.supports_microversion', autospec=True, return_value=False)
@mock.patch.object(resource.Resource, '_translate_response')
def test_create_pre_v355(self, mock_mv, mock_translate):
    self.sess.default_microversion = '3.0'
    sot = transfer.Transfer()
    sot.create(self.sess, volume_id=FAKE_VOL_ID, name=FAKE_VOL_NAME)
    self.sess.post.assert_called_with('/os-volume-transfer', json={'transfer': {}}, microversion='3.0', headers={}, params={'volume_id': FAKE_VOL_ID, 'name': FAKE_VOL_NAME})