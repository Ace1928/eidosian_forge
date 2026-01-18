from oslo_config import fixture
from oslotest import base as test_base
import webob
import webob.dec
import webob.exc as exc
from oslo_middleware import cors
def test_invalid_default_option(self):
    """Assert that using set_defaults only permits valid options."""
    self.assertRaises(AttributeError, cors.set_defaults, allowed_origin='test')