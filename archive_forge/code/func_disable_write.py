import fixtures
from keystone.identity.backends.ldap import common as common_ldap
from keystone.tests.unit import fakeldap
def disable_write(self):
    common_ldap.WRITABLE = False