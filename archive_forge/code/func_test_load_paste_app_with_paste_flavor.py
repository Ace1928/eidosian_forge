import os.path
import shutil
import fixtures
import oslo_middleware
from glance.api.middleware import context
from glance.common import config
from glance.tests import utils as test_utils
def test_load_paste_app_with_paste_flavor(self):
    pipeline = '[composite:glance-api-incomplete]\npaste.composite_factory = glance.api:root_app_factory\n/: api-incomplete\n/healthcheck: healthcheck\n[pipeline:api-incomplete]\npipeline = context rootapp'
    expected_middleware = context.ContextMiddleware
    self._do_test_load_paste_app(expected_middleware, paste_flavor='incomplete', paste_append=pipeline)