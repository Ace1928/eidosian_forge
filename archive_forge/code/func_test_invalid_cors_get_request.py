import http.client
import httplib2
from oslo_utils.fixture import uuidsentinel as uuids
from glance.tests import functional
def test_invalid_cors_get_request(self):
    r_headers, content = self.http.request(self.api_path, 'GET', headers=self._headers({'Origin': 'http://invalid.example.com'}))
    self.assertEqual(http.client.OK, r_headers.status)
    self.assertNotIn('access-control-allow-origin', r_headers)