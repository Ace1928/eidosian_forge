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
def test_dn_parsed_dns(self):
    dn_str1 = ldap.dn.str2dn('cn=Babs Jansen,ou=OpenStack+cn=OpenSource')
    dn_str2 = ldap.dn.str2dn('CN=Babs Jansen,cn=OpenSource+ou=OpenStack')
    self.assertTrue(common_ldap.is_dn_equal(dn_str1, dn_str2))