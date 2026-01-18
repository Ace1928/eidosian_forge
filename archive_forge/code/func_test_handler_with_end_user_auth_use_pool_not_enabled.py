from unittest import mock
import fixtures
import ldappool
from keystone.common import provider_api
import keystone.conf
from keystone.identity.backends import ldap
from keystone.identity.backends.ldap import common as common_ldap
from keystone.tests import unit
from keystone.tests.unit import fakeldap
from keystone.tests.unit import test_backend_ldap
@mock.patch.object(common_ldap.KeystoneLDAPHandler, 'connect')
@mock.patch.object(common_ldap.KeystoneLDAPHandler, 'simple_bind_s')
def test_handler_with_end_user_auth_use_pool_not_enabled(self, bind_method, connect_method):
    self.config_fixture.config(group='ldap', use_auth_pool=False)
    self.cleanup_pools()
    user_api = ldap.UserApi(CONF)
    handler = user_api.get_connection(user=None, password=None, end_user_auth=True)
    self.assertIsInstance(handler.conn, common_ldap.PythonLDAPHandler)
    handler = user_api.get_connection(user=None, password=None, end_user_auth=False)
    self.assertIsInstance(handler.conn, common_ldap.PooledLDAPHandler)