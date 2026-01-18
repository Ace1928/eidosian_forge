from unittest import mock
import oslo_policy.policy
from oslo_utils.fixture import uuidsentinel as uuids
from glance.api import policy
from glance.tests import functional
def test_image_reactivate(self):
    self.start_server()
    image_id = self._create_and_upload()
    resp = self.api_post('/v2/images/%s/actions/deactivate' % image_id)
    self.assertEqual(204, resp.status_code)
    resp = self.api_get('/v2/images/%s' % image_id)
    self.assertEqual('deactivated', resp.json['status'])
    resp = self.api_post('/v2/images/%s/actions/reactivate' % image_id)
    self.assertEqual(204, resp.status_code)
    resp = self.api_get('/v2/images/%s' % image_id)
    self.assertEqual('active', resp.json['status'])
    resp = self.api_post('/v2/images/%s/actions/deactivate' % image_id)
    self.assertEqual(204, resp.status_code)
    self.set_policy_rules({'get_image': '', 'reactivate': '!'})
    resp = self.api_post('/v2/images/%s/actions/reactivate' % image_id)
    self.assertEqual(403, resp.status_code)
    self.set_policy_rules({'get_image': '!', 'reactivate': '!'})
    resp = self.api_post('/v2/images/%s/actions/reactivate' % image_id)
    self.assertEqual(404, resp.status_code)
    self.set_policy_rules({'get_image': '!', 'reactivate': ''})
    resp = self.api_post('/v2/images/%s/actions/reactivate' % image_id)
    self.assertEqual(204, resp.status_code)
    self.set_policy_rules({'get_image': '', 'modify_image': '', 'add_image': '', 'upload_image': '', 'add_member': '', 'deactivate': '', 'reactivate': '', 'publicize_image': '', 'communitize_image': ''})
    headers = self._headers({'X-Project-Id': 'fake-project-id', 'X-Roles': 'member'})
    for visibility in ('public', 'community', 'shared', 'private'):
        image_id = self._create_and_upload(visibility=visibility)
        resp = self.api_post('/v2/images/%s/actions/deactivate' % image_id)
        self.assertEqual(204, resp.status_code)
        resp = self.api_post('/v2/images/%s/actions/reactivate' % image_id, headers=headers)
        if visibility == 'shared':
            self.assertEqual(404, resp.status_code)
            share_path = '/v2/images/%s/members' % image_id
            data = {'member': 'fake-project-id'}
            response = self.api_post(share_path, json=data)
            member = response.json
            self.assertEqual(200, response.status_code)
            self.assertEqual(image_id, member['image_id'])
            resp = self.api_post('/v2/images/%s/actions/reactivate' % image_id, headers=headers)
            self.assertEqual(403, resp.status_code)
        elif visibility == 'private':
            self.assertEqual(404, resp.status_code)
        else:
            self.assertEqual(403, resp.status_code)