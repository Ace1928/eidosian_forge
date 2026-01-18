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
def test_filter_with_hints_and_query_is_none(self):
    hints = driver_hints.Hints()
    username = uuid.uuid4().hex
    hints.add_filter(name=self.attribute_name, value=username, comparator='equals', case_sensitive=False)
    expected_ldap_filter = '(&(%s=%s))' % (self.filter_attribute_name, username)
    self.assertEqual(expected_ldap_filter, self.base_ldap.filter_query(hints=hints, query=None))