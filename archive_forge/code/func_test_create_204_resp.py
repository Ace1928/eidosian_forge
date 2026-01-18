import copy
from unittest import mock
import uuid
from openstack.message.v2 import claim
from openstack.tests.unit import base
@mock.patch.object(uuid, 'uuid4')
def test_create_204_resp(self, mock_uuid):
    sess = mock.Mock()
    resp = mock.Mock()
    sess.post.return_value = resp
    resp.status_code = 204
    sess.get_project_id.return_value = 'NEW_PROJECT_ID'
    mock_uuid.return_value = 'NEW_CLIENT_ID'
    FAKE = copy.deepcopy(FAKE1)
    sot = claim.Claim(**FAKE1)
    res = sot.create(sess)
    url = '/queues/%(queue)s/claims' % {'queue': FAKE.pop('queue_name')}
    headers = {'Client-ID': 'NEW_CLIENT_ID', 'X-PROJECT-ID': 'NEW_PROJECT_ID'}
    sess.post.assert_called_once_with(url, headers=headers, json=FAKE)
    sess.get_project_id.assert_called_once_with()
    self.assertEqual(sot, res)