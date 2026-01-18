from unittest import mock
import oslo_policy.policy
from oslo_utils.fixture import uuidsentinel as uuids
from glance.api import policy
from glance.tests import functional
def test_member_get_basic(self):
    self.start_server()
    output = self.load_data(share_image=True)
    path = '/v2/images/%s/members/%s' % (output['image_id'], output['member_id'])
    response = self.api_get(path)
    self.assertEqual(200, response.status_code)
    member = response.json
    self.assertEqual(output['image_id'], member['image_id'])
    self.assertEqual('pending', member['status'])
    self.set_policy_rules({'get_member': '!'})
    response = self.api_get(path)
    self.assertEqual(404, response.status_code)