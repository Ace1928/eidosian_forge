import uuid
from keystoneauth1.exceptions import http
from keystoneclient import exceptions
from keystoneclient.tests.functional import base
from keystoneclient.tests.functional.v3 import client_fixtures as fixtures
def test_grant_role_invalid_params(self):
    user = fixtures.User(self.client, self.project_domain_id)
    self.useFixture(user)
    role = fixtures.Role(self.client, domain=self.project_domain_id)
    self.useFixture(role)
    self.assertRaises(exceptions.ValidationError, self.client.roles.grant, role.id, user=user.id)
    group = fixtures.Group(self.client, self.project_domain_id)
    self.useFixture(group)
    self.assertRaises(exceptions.ValidationError, self.client.roles.grant, role.id, group=group.id)