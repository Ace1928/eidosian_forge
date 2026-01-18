import uuid
from unittest import mock
from keystone.assignment.core import Manager as AssignmentApi
from keystone.auth.plugins import mapped
from keystone.exception import ProjectNotFound
from keystone.resource.core import Manager as ResourceApi
from keystone.tests import unit
def test_configure_project_domain_with_domain_name(self):
    domain_name = 'test-domain'
    self.shadow_project_mock['domain'] = {'name': domain_name}
    self.resource_api_mock.get_domain_by_name.return_value = self.domain_mock
    mapped.configure_project_domain(self.shadow_project_mock, self.idp_domain_uuid_mock, self.resource_api_mock)
    self.assertIn('domain', self.shadow_project_mock)
    self.assertEqual(self.domain_uuid_mock, self.shadow_project_mock['domain']['id'])
    self.resource_api_mock.get_domain_by_name.assert_called_with(domain_name)