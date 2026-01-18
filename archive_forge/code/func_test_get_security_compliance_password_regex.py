import copy
import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_get_security_compliance_password_regex(self):
    """Ask for the security compliance password regular expression."""
    password_regex = uuid.uuid4().hex
    self.config_fixture.config(group='security_compliance', password_regex=password_regex)
    group = 'security_compliance'
    option = 'password_regex'
    url = '/domains/%(domain_id)s/config/%(group)s/%(option)s' % {'domain_id': CONF.identity.default_domain_id, 'group': group, 'option': option}
    regular_response = self.get(url, token=self._get_non_admin_token())
    self.assertEqual(regular_response.result['config'][option], password_regex)
    admin_response = self.get(url, token=self._get_admin_token())
    self.assertEqual(admin_response.result['config'][option], password_regex)
    self.head(url, token=self._get_non_admin_token(), expected_status=http.client.OK)
    self.head(url, token=self._get_admin_token(), expected_status=http.client.OK)