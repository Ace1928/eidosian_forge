import ldap
from keystone.common import cache
from keystone.common import provider_api
import keystone.conf
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit.ksfixtures import database
from keystone.tests.unit.ksfixtures import ldapdb
def get_user_enabled_vals(self, user):
    user_dn = PROVIDERS.identity_api.driver.user._id_to_dn_string(user['id'])
    enabled_attr_name = CONF.ldap.user_enabled_attribute
    ldap_ = PROVIDERS.identity_api.driver.user.get_connection()
    res = ldap_.search_s(user_dn, ldap.SCOPE_BASE, u'(sn=%s)' % user['name'])
    if enabled_attr_name in res[0][1]:
        return res[0][1][enabled_attr_name]
    else:
        return None