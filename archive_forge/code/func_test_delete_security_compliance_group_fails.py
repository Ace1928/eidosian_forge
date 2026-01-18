import copy
import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_delete_security_compliance_group_fails(self):
    """The security compliance group shouldn't be deleteable."""
    url = '/domains/%(domain_id)s/config/%(group)s/' % {'domain_id': CONF.identity.default_domain_id, 'group': 'security_compliance'}
    self.delete(url, expected_status=http.client.FORBIDDEN, token=self._get_non_admin_token())
    self.delete(url, expected_status=http.client.FORBIDDEN, token=self._get_admin_token())