from keystoneauth1.exceptions import http
from keystoneclient.tests.functional import base
from keystoneclient.tests.functional.v3 import client_fixtures as fixtures
def test_create_ec2(self):
    user = fixtures.User(self.client, self.project_domain_id)
    self.useFixture(user)
    project = fixtures.Project(self.client, self.project_domain_id)
    self.useFixture(project)
    ec2_ref = {'user_id': user.id, 'project_id': project.id}
    ec2 = self.client.ec2.create(**ec2_ref)
    self.addCleanup(self.client.ec2.delete, user.id, ec2.access)
    self.check_ec2(ec2, ec2_ref)