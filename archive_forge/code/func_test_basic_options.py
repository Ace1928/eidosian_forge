from keystoneauth1 import http_basic
from keystoneauth1.loading._plugins import http_basic as loader
from keystoneauth1 import session
from keystoneauth1.tests.unit import utils
def test_basic_options(self):
    opts = loader.HTTPBasicAuth().get_options()
    self.assertEqual(['username', 'password', 'endpoint'], [o.name for o in opts])