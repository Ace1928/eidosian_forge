import copy
import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_get_head_config_default_for_unsupported_group(self):
    self.get('/domains/config/ldap/password/default', expected_status=http.client.FORBIDDEN)
    self.head('/domains/config/ldap/password/default', expected_status=http.client.FORBIDDEN)