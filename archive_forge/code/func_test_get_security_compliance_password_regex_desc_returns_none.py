import copy
import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_get_security_compliance_password_regex_desc_returns_none(self):
    """When an option isn't set, we should explicitly return None."""
    group = 'security_compliance'
    option = 'password_regex_description'
    url = '/domains/%(domain_id)s/config/%(group)s/%(option)s' % {'domain_id': CONF.identity.default_domain_id, 'group': group, 'option': option}
    regular_response = self.get(url, token=self._get_non_admin_token())
    self.assertIsNone(regular_response.result['config'][option])
    admin_response = self.get(url, token=self._get_admin_token())
    self.assertIsNone(admin_response.result['config'][option])
    self.head(url, token=self._get_non_admin_token(), expected_status=http.client.OK)
    self.head(url, token=self._get_admin_token(), expected_status=http.client.OK)