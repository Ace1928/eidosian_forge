import copy
from unittest import mock
from keystoneauth1 import adapter
from openstack.block_storage.v3 import snapshot
from openstack.tests.unit import base
@mock.patch('openstack.utils.supports_microversion', autospec=True, return_value=False)
def test_manage_pre_38(self, mock_mv):
    resp = mock.Mock()
    resp.body = {'snapshot': copy.deepcopy(SNAPSHOT)}
    resp.json = mock.Mock(return_value=resp.body)
    resp.headers = {}
    resp.status_code = 202
    self.sess.post = mock.Mock(return_value=resp)
    sot = snapshot.Snapshot.manage(self.sess, volume_id=FAKE_VOLUME_ID, ref=FAKE_ID)
    self.assertIsNotNone(sot)
    url = '/os-snapshot-manage'
    body = {'snapshot': {'volume_id': FAKE_VOLUME_ID, 'ref': FAKE_ID, 'name': None, 'description': None, 'metadata': None}}
    self.sess.post.assert_called_with(url, json=body, microversion=sot._max_microversion)