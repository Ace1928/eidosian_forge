import os
import tempfile
from unittest import mock
import uuid
import fixtures
import ldap.dn
from oslo_config import fixture as config_fixture
from keystone.common import driver_hints
from keystone.common import provider_api
import keystone.conf
from keystone import exception as ks_exception
from keystone.identity.backends.ldap import common as common_ldap
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit import fakeldap
from keystone.tests.unit.ksfixtures import database
from keystone.tests.unit.ksfixtures import ldapdb
@mock.patch.object(common_ldap.KeystoneLDAPHandler, 'simple_bind_s')
def test_multiple_urls_with_comma_no_conn_pool(self, mock_ldap_bind):
    urls = 'ldap://localhost,ldap://backup.localhost'
    self.config_fixture.config(group='ldap', url=urls, use_pool=False)
    base_ldap = common_ldap.BaseLdap(CONF)
    ldap_connection = base_ldap.get_connection()
    self.assertEqual(urls, ldap_connection.conn.conn._uri)