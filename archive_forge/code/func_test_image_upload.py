from unittest import mock
import oslo_policy.policy
from oslo_utils.fixture import uuidsentinel as uuids
from glance.api import policy
from glance.tests import functional
def test_image_upload(self):
    self.start_server()
    self._create_and_upload(expected_code=204)
    self.set_policy_rules({'add_image': '', 'get_image': '', 'upload_image': '!'})
    self._create_and_upload(expected_code=403)
    self.set_policy_rules({'add_image': '', 'get_image': '!', 'upload_image': '!'})
    self._create_and_upload(expected_code=404)
    self.set_policy_rules({'add_image': '', 'get_image': '!', 'upload_image': ''})
    self._create_and_upload(expected_code=204)