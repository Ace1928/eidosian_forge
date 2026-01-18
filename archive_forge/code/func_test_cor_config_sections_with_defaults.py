from oslo_config import fixture
from oslotest import base as test_base
import webob
import webob.dec
import webob.exc as exc
from oslo_middleware import cors
def test_cor_config_sections_with_defaults(self):
    """Assert cors.* config sections with default values work."""
    self.config_fixture.load_raw_values(group='cors.subdomain')
    self.application = cors.CORS(test_application, self.config)