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
def test_max_connection_error_raised(self):
    who = CONF.ldap.user
    cred = CONF.ldap.password
    ldappool_cm = self.conn_pools[CONF.ldap.url]
    ldappool_cm.size = 2
    with ldappool_cm.connection(who, cred) as _:
        with ldappool_cm.connection(who, cred) as _:
            try:
                with ldappool_cm.connection(who, cred) as _:
                    _.unbind_s()
                    self.fail()
            except Exception as ex:
                self.assertIsInstance(ex, ldappool.MaxConnectionReachedError)
    ldappool_cm.size = CONF.ldap.pool_size