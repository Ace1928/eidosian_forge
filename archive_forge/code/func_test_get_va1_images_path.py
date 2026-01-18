import http.client as http_client
import httplib2
from oslo_serialization import jsonutils
from glance.tests import functional
from glance.tests.unit import test_versions as tv
def test_get_va1_images_path(self):
    """Assert GET /va.1/images with no Accept: header
        Verify version choices returned
        """
    path = 'http://%s:%d/va.1/images' % ('127.0.0.1', self.api_port)
    http = httplib2.Http()
    response, content_json = http.request(path, 'GET')
    self.assertEqual(http_client.MULTIPLE_CHOICES, response.status)
    content = jsonutils.loads(content_json.decode())
    self.assertEqual(self.versions, content)