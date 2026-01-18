import http.client as http_client
import httplib2
from oslo_serialization import jsonutils
from glance.tests import functional
from glance.tests.unit import test_versions as tv
def test_get_root_path(self):
    """Assert GET / with `no Accept:` header.
        Verify version choices returned.
        Bug lp:803260  no Accept header causes a 500 in glance-api
        """
    path = 'http://%s:%d' % ('127.0.0.1', self.api_port)
    http = httplib2.Http()
    response, content_json = http.request(path, 'GET')
    self.assertEqual(http_client.MULTIPLE_CHOICES, response.status)
    content = jsonutils.loads(content_json.decode())
    self.assertEqual(self.versions, content)