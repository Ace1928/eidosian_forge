import http.client
import tempfile
import httplib2
from glance.tests import functional
from glance.tests import utils
def test_healthcheck_disabled(self):
    with tempfile.NamedTemporaryFile() as test_disable_file:
        self.cleanup()
        self.api_server.disable_path = test_disable_file.name
        self.start_servers(**self.__dict__.copy())
        response, content = self.request()
        self.assertEqual(b'DISABLED BY FILE', content)
        self.assertEqual(http.client.SERVICE_UNAVAILABLE, response.status)
        self.stop_servers()