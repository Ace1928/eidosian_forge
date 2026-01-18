from unittest import mock
import oslo_policy.policy
from oslo_utils.fixture import uuidsentinel as uuids
from glance.api import policy
from glance.tests import functional
def test_image_update_basic(self):
    self.start_server()
    image_id = self._create_and_upload()
    resp = self.api_patch('/v2/images/%s' % image_id, {'op': 'add', 'path': '/mykey1', 'value': 'foo'})
    self.assertEqual(200, resp.status_code, resp.text)
    self.assertEqual('foo', self.api_get('/v2/images/%s' % image_id).json['mykey1'])
    self.set_policy_rules({'get_image': '', 'modify_image': '!'})
    resp = self.api_patch('/v2/images/%s' % image_id, {'op': 'add', 'path': '/mykey2', 'value': 'foo'})
    self.assertEqual(403, resp.status_code)
    self.assertNotIn('mykey2', self.api_get('/v2/images/%s' % image_id).json)
    resp = self.api_patch('/v2/images/%s' % image_id, {'op': 'replace', 'path': '/mykey1', 'value': 'bar'})
    self.assertEqual(403, resp.status_code)
    self.assertEqual('foo', self.api_get('/v2/images/%s' % image_id).json['mykey1'])
    resp = self.api_patch('/v2/images/%s' % image_id, {'op': 'remove', 'path': '/mykey1'})
    self.assertEqual(403, resp.status_code)
    self.assertEqual('foo', self.api_get('/v2/images/%s' % image_id).json['mykey1'])
    self.set_policy_rules({'get_image': '!', 'modify_image': '!'})
    resp = self.api_patch('/v2/images/%s' % image_id, {'op': 'remove', 'path': '/mykey1'})
    self.assertEqual(404, resp.status_code)