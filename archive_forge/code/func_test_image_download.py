from unittest import mock
import oslo_policy.policy
from oslo_utils.fixture import uuidsentinel as uuids
from glance.api import policy
from glance.tests import functional
def test_image_download(self):
    self.start_server()
    image_id = self._create_and_upload()
    path = '/v2/images/%s/file' % image_id
    response = self.api_get(path)
    self.assertEqual(200, response.status_code)
    self.assertEqual('IMAGEDATA', response.text)
    self.set_policy_rules({'get_image': '', 'download_image': '!'})
    response = self.api_get(path)
    self.assertEqual(403, response.status_code)
    self.set_policy_rules({'get_image': '!', 'download_image': '!'})
    response = self.api_get(path)
    self.assertEqual(404, response.status_code)
    self.set_policy_rules({'get_image': '!', 'download_image': ''})
    response = self.api_get(path)
    self.assertEqual(200, response.status_code)
    self.assertEqual('IMAGEDATA', response.text)