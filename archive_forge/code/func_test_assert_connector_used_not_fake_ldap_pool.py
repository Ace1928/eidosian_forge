import ldappool
from keystone.common import provider_api
import keystone.conf
from keystone.identity.backends import ldap
from keystone.identity.backends.ldap import common as ldap_common
from keystone.tests import unit
from keystone.tests.unit import fakeldap
from keystone.tests.unit import test_backend_ldap_pool
from keystone.tests.unit import test_ldap_livetest
def test_assert_connector_used_not_fake_ldap_pool(self):
    handler = ldap_common._get_connection(CONF.ldap.url, use_pool=True)
    self.assertNotEqual(type(handler.Connector), type(fakeldap.FakeLdapPool))
    self.assertEqual(type(ldappool.StateConnector), type(handler.Connector))