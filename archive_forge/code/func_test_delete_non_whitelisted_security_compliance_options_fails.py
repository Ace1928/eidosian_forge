import copy
import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_delete_non_whitelisted_security_compliance_options_fails(self):
    """The security compliance options shouldn't be deleteable."""
    url = '/domains/%(domain_id)s/config/%(group)s/%(option)s' % {'domain_id': CONF.identity.default_domain_id, 'group': 'security_compliance', 'option': 'lockout_failure_attempts'}
    self.delete(url, expected_status=http.client.FORBIDDEN, token=self._get_non_admin_token())
    self.delete(url, expected_status=http.client.FORBIDDEN, token=self._get_admin_token())