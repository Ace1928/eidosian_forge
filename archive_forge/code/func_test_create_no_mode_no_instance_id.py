from unittest import mock
from keystoneauth1 import adapter
from openstack.block_storage.v3 import attachment
from openstack import resource
from openstack.tests.unit import base
@mock.patch('openstack.utils.supports_microversion', autospec=True, return_value=True)
@mock.patch.object(resource.Resource, '_translate_response')
def test_create_no_mode_no_instance_id(self, mock_translate, mock_mv):
    self.sess.default_microversion = '3.27'
    mock_mv.return_value = False
    sot = attachment.Attachment()
    FAKE_MODE = 'rw'
    sot.create(self.sess, volume_id=FAKE_VOL_ID, connector=CONNECTOR, instance=None, mode=FAKE_MODE)
    self.sess.post.assert_called_with('/attachments', json={'attachment': {}}, headers={}, microversion='3.27', params={'volume_id': FAKE_VOL_ID, 'connector': CONNECTOR, 'instance': None, 'mode': 'rw'})
    self.sess.default_microversion = '3.54'