import uuid
from keystoneauth1.exceptions import http
from keystoneclient import exceptions
from keystoneclient.tests.functional import base
from keystoneclient.tests.functional.v3 import client_fixtures as fixtures
def test_group_domain_grant_and_revoke(self):
    group = fixtures.Group(self.client, self.project_domain_id)
    self.useFixture(group)
    domain = fixtures.Domain(self.client)
    self.useFixture(domain)
    role = fixtures.Role(self.client, domain=self.project_domain_id)
    self.useFixture(role)
    self.client.roles.grant(role, group=group.id, domain=domain.id)
    roles_after_grant = self.client.roles.list(group=group.id, domain=domain.id)
    self.assertCountEqual(roles_after_grant, [role.entity])
    self.client.roles.revoke(role, group=group.id, domain=domain.id)
    roles_after_revoke = self.client.roles.list(group=group.id, domain=domain.id)
    self.assertEqual(roles_after_revoke, [])