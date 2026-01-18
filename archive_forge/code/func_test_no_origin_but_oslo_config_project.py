from oslo_config import fixture
from oslotest import base as test_base
import webob
import webob.dec
import webob.exc as exc
from oslo_middleware import cors
def test_no_origin_but_oslo_config_project(self):
    """Assert that a filter factory with oslo_config_project succeed."""
    cors.filter_factory(global_conf=None, oslo_config_project='foobar')