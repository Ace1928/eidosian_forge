from unittest import mock
from keystoneauth1 import adapter
from openstack.block_storage.v3 import transfer
from openstack import resource
from openstack.tests.unit import base
@mock.patch('openstack.utils.supports_microversion', autospec=True, return_value=False)
@mock.patch.object(resource.Resource, '_translate_response')
def test_accept_pre_v355(self, mock_mv, mock_translate):
    self.sess.default_microversion = '3.0'
    sot = transfer.Transfer()
    sot.id = FAKE_TRANSFER
    sot.accept(self.sess, auth_key=FAKE_AUTH_KEY)
    self.sess.post.assert_called_with('os-volume-transfer/%s/accept' % FAKE_TRANSFER, json={'accept': {'auth_key': FAKE_AUTH_KEY}}, microversion='3.0')