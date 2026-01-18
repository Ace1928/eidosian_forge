from unittest import mock
import oslo_policy.policy
from oslo_utils.fixture import uuidsentinel as uuids
from glance.api import policy
from glance.tests import functional
def test_image_sharing_not_allowed(self):
    self.start_server()
    path = '/v2/images'
    for visibility in ('community', 'private', 'public'):
        data = {'name': '%s-image' % visibility, 'visibility': visibility}
        response = self.api_post(path, json=data)
        image = response.json
        self.assertEqual(201, response.status_code)
        self.assertEqual(visibility, image['visibility'])
        member_path = '/v2/images/%s/members' % image['id']
        data = {'member': uuids.random_member}
        response = self.api_post(member_path, json=data)
        self.assertEqual(403, response.status_code)