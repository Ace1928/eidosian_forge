from keystoneauth1.exceptions import http
from keystoneclient.tests.functional import base
from keystoneclient.tests.functional.v3 import client_fixtures as fixtures
def test_get_ec2(self):
    user = fixtures.User(self.client, self.project_domain_id)
    self.useFixture(user)
    project = fixtures.Project(self.client, self.project_domain_id)
    self.useFixture(project)
    ec2 = fixtures.EC2(self.client, user_id=user.id, project_id=project.id)
    self.useFixture(ec2)
    ec2_ret = self.client.ec2.get(user.id, ec2.access)
    self.check_ec2(ec2_ret, ec2.ref)