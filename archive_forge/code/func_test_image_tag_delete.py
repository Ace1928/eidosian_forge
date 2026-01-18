from unittest import mock
import oslo_policy.policy
from oslo_utils.fixture import uuidsentinel as uuids
from glance.api import policy
from glance.tests import functional
def test_image_tag_delete(self):
    self.start_server()
    image_id = self._create_and_upload()
    path = '/v2/images/%s/tags/Test_Tag_1' % image_id
    response = self.api_put(path)
    self.assertEqual(204, response.status_code)
    path = '/v2/images/%s/tags/Test_Tag_2' % image_id
    response = self.api_put(path)
    self.assertEqual(204, response.status_code)
    path = '/v2/images/%s' % image_id
    response = self.api_get(path)
    image = response.json
    self.assertItemsEqual(['Test_Tag_1', 'Test_Tag_2'], image['tags'])
    path = '/v2/images/%s/tags/Test_Tag_1' % image_id
    response = self.api_delete(path)
    self.assertEqual(204, response.status_code)
    path = '/v2/images/%s' % image_id
    response = self.api_get(path)
    image = response.json
    self.assertNotIn('Test_Tag_1', image['tags'])
    self.set_policy_rules({'get_image': '!', 'modify_image': '!'})
    path = '/v2/images/%s/tags/Test_Tag_2' % image_id
    response = self.api_delete(path)
    self.assertEqual(404, response.status_code)
    self.set_policy_rules({'get_image': '', 'modify_image': '!'})
    path = '/v2/images/%s/tags/Test_Tag_2' % image_id
    response = self.api_delete(path)
    self.assertEqual(403, response.status_code)
    self.set_policy_rules({'get_image': '', 'modify_image': ''})
    headers = self._headers({'X-Project-Id': 'fake-project-id', 'X-Roles': 'member'})
    path = '/v2/images/%s/tags/Test_Tag_2' % image_id
    response = self.api_delete(path, headers=headers)
    self.assertEqual(404, response.status_code)
    self._test_image_ownership(headers, 'DELETE')