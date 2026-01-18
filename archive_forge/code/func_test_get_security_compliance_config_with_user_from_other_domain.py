import copy
import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_get_security_compliance_config_with_user_from_other_domain(self):
    """Make sure users from other domains can access password requirements.

        Even though a user is in a separate domain, they should be able to see
        the security requirements for the deployment. This is because security
        compliance is not yet implemented on a per domain basis. Once that
        happens, then this should no longer be possible since a user should
        only care about the security compliance requirements for the domain
        that they are in.
        """
    domain = unit.new_domain_ref()
    PROVIDERS.resource_api.create_domain(domain['id'], domain)
    user = unit.create_user(PROVIDERS.identity_api, domain['id'])
    project = unit.new_project_ref(domain_id=domain['id'])
    PROVIDERS.resource_api.create_project(project['id'], project)
    PROVIDERS.assignment_api.add_role_to_user_and_project(user['id'], project['id'], self.non_admin_role['id'])
    password_regex = uuid.uuid4().hex
    password_regex_description = uuid.uuid4().hex
    group = 'security_compliance'
    self.config_fixture.config(group=group, password_regex=password_regex)
    self.config_fixture.config(group=group, password_regex_description=password_regex_description)
    user_token = self.build_authentication_request(user_id=user['id'], password=user['password'], project_id=project['id'])
    user_token = self.get_requested_token(user_token)
    url = '/domains/%(domain_id)s/config/%(group)s' % {'domain_id': CONF.identity.default_domain_id, 'group': group}
    response = self.get(url, token=user_token)
    self.assertEqual(response.result['config'][group]['password_regex'], password_regex)
    self.assertEqual(response.result['config'][group]['password_regex_description'], password_regex_description)
    self.head(url, token=user_token, expected_status=http.client.OK)