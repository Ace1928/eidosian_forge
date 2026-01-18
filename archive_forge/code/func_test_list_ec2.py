from keystoneauth1.exceptions import http
from keystoneclient.tests.functional import base
from keystoneclient.tests.functional.v3 import client_fixtures as fixtures
def test_list_ec2(self):
    user_one = fixtures.User(self.client, self.project_domain_id)
    self.useFixture(user_one)
    ec2_one = fixtures.EC2(self.client, user_id=user_one.id, project_id=self.project_domain_id)
    self.useFixture(ec2_one)
    user_two = fixtures.User(self.client, self.project_domain_id)
    self.useFixture(user_two)
    ec2_two = fixtures.EC2(self.client, user_id=user_two.id, project_id=self.project_domain_id)
    self.useFixture(ec2_two)
    ec2_list = self.client.ec2.list(user_one.id)
    for ec2 in ec2_list:
        self.check_ec2(ec2)
    self.assertIn(ec2_one.entity, ec2_list)
    self.assertNotIn(ec2_two.entity, ec2_list)