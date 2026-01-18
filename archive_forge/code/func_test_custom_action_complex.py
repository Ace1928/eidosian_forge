import uuid
from pycadf import cadftaxonomy as taxonomy
import webob
from keystonemiddleware import audit
from keystonemiddleware.tests.unit.audit import base
def test_custom_action_complex(self):
    url = 'http://admin_host:8774/v2/%s/os-migrations' % uuid.uuid4().hex
    payload = self.get_payload('GET', url)
    self.assertEqual(payload['target']['typeURI'], 'service/compute/os-migrations')
    self.assertEqual(payload['action'], 'read')
    payload = self.get_payload('POST', url)
    self.assertEqual(payload['target']['typeURI'], 'service/compute/os-migrations')
    self.assertEqual(payload['action'], 'create')