from keystoneauth1.exceptions import http
from keystoneclient.tests.functional import base
from keystoneclient.tests.functional.v3 import client_fixtures as fixtures
def check_ec2(self, ec2, ec2_ref=None):
    self.assertIn('self', ec2.links)
    self.assertIn('/users/%s/credentials/OS-EC2/%s' % (ec2.user_id, ec2.access), ec2.links['self'])
    if ec2_ref:
        self.assertEqual(ec2_ref['user_id'], ec2.user_id)
        self.assertEqual(ec2_ref['project_id'], ec2.tenant_id)
    else:
        self.assertIsNotNone(ec2.user_id)
        self.assertIsNotNone(ec2.tenant_id)