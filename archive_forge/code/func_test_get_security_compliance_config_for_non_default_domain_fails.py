import copy
import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_get_security_compliance_config_for_non_default_domain_fails(self):
    """Getting security compliance opts for other domains should fail.

        Support for enforcing security compliance rules per domain currently
        does not exist, so exposing security compliance information for any
        domain other than the default domain should not be allowed.
        """
    domain = unit.new_domain_ref()
    PROVIDERS.resource_api.create_domain(domain['id'], domain)
    password_regex = uuid.uuid4().hex
    password_regex_description = uuid.uuid4().hex
    self.config_fixture.config(group='security_compliance', password_regex=password_regex)
    self.config_fixture.config(group='security_compliance', password_regex_description=password_regex_description)
    url = '/domains/%(domain_id)s/config/%(group)s' % {'domain_id': domain['id'], 'group': 'security_compliance'}
    self.get(url, expected_status=http.client.FORBIDDEN, token=self._get_non_admin_token())
    self.get(url, expected_status=http.client.FORBIDDEN, token=self._get_admin_token())
    self.head(url, expected_status=http.client.FORBIDDEN, token=self._get_non_admin_token())
    self.head(url, expected_status=http.client.FORBIDDEN, token=self._get_admin_token())