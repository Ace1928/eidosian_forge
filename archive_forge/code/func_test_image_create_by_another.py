from unittest import mock
import oslo_policy.policy
from oslo_utils.fixture import uuidsentinel as uuids
from glance.api import policy
from glance.tests import functional
def test_image_create_by_another(self):
    self.start_server()
    image = {'name': 'foo', 'container_format': 'bare', 'disk_format': 'raw', 'owner': 'someoneelse'}
    resp = self.api_post('/v2/images', json=image, headers={'X-Roles': 'member'})
    self.assertIn("You are not permitted to create images owned by 'someoneelse'", resp.text)