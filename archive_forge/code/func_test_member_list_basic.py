from unittest import mock
import oslo_policy.policy
from oslo_utils.fixture import uuidsentinel as uuids
from glance.api import policy
from glance.tests import functional
def test_member_list_basic(self):
    self.start_server()
    output = self.load_data(share_image=True)
    path = '/v2/images/%s/members' % output['image_id']
    response = self.api_get(path)
    self.assertEqual(200, response.status_code)
    self.assertEqual(1, len(response.json['members']))
    self.set_policy_rules({'get_members': '!', 'get_image': '@'})
    response = self.api_get(path)
    self.assertEqual(403, response.status_code)
    self.set_policy_rules({'get_members': '!', 'get_image': '!'})
    response = self.api_get(path)
    self.assertEqual(404, response.status_code)
    self.set_policy_rules({'get_members': '@', 'get_member': '!', 'get_image': '@'})
    response = self.api_get(path)
    self.assertEqual(200, response.status_code)
    self.assertEqual(0, len(response.json['members']))