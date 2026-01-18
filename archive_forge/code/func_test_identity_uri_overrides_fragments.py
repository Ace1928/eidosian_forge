import io
import uuid
from keystoneauth1 import fixture
from keystoneauth1 import plugin as ksa_plugin
from keystoneauth1 import session
from oslo_log import log as logging
from requests_mock.contrib import fixture as rm_fixture
from keystonemiddleware.auth_token import _auth
from keystonemiddleware.tests.unit import utils
def test_identity_uri_overrides_fragments(self):
    identity_uri = 'http://testhost:8888/admin'
    plugin = self.new_plugin(identity_uri=identity_uri, auth_host='anotherhost', auth_port=9999, auth_protocol='ftp')
    endpoint = plugin.get_endpoint(self.session, interface=ksa_plugin.AUTH_INTERFACE)
    self.assertEqual(identity_uri, endpoint)