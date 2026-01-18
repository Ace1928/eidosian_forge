from unittest import mock
import oslo_policy.policy
from oslo_utils.fixture import uuidsentinel as uuids
from glance.api import policy
from glance.tests import functional
def test_image_get(self):
    self.start_server()
    image_id = self._create_and_upload()
    image = self.api_get('/v2/images/%s' % image_id).json
    self.assertEqual(image_id, image['id'])
    images = self.api_get('/v2/images').json['images']
    self.assertEqual(1, len(images))
    self.assertEqual(image_id, images[0]['id'])
    self.set_policy_rules({'get_images': '!', 'get_image': ''})
    resp = self.api_get('/v2/images')
    self.assertEqual(403, resp.status_code)
    image = self.api_get('/v2/images/%s' % image_id).json
    self.assertEqual(image_id, image['id'])
    self.set_policy_rules({'get_images': '', 'get_image': '!'})
    images = self.api_get('/v2/images').json['images']
    self.assertEqual(0, len(images))
    resp = self.api_get('/v2/images/%s' % image_id)
    self.assertEqual(404, resp.status_code)
    self.set_policy_rules({'get_images': '!', 'get_image': '!'})
    resp = self.api_get('/v2/images')
    self.assertEqual(403, resp.status_code)
    resp = self.api_get('/v2/images/%s' % image_id)
    self.assertEqual(404, resp.status_code)