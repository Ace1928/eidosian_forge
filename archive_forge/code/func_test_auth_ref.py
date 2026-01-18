import uuid
from keystoneauth1 import access
from keystoneauth1 import fixture
from keystoneauth1.identity import access as access_plugin
from keystoneauth1 import plugin
from keystoneauth1 import session
from keystoneauth1.tests.unit import utils
def test_auth_ref(self):
    plugin_obj = self._plugin()
    self.assertEqual(self.TEST_ROOT_URL, plugin_obj.get_endpoint(self.session, service_type='identity', interface='public'))
    self.assertEqual(self.auth_token, plugin_obj.get_token(session))