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
def test_filter_with_both_query_and_hints_set(self):
    hints = driver_hints.Hints()
    query = uuid.uuid4().hex
    username = uuid.uuid4().hex
    expected_result = '(&%(query)s(%(user_name_attr)s=%(username)s))' % {'query': query, 'user_name_attr': self.filter_attribute_name, 'username': username}
    hints.add_filter(self.attribute_name, username)
    self.assertEqual(expected_result, self.base_ldap.filter_query(hints=hints, query=query))