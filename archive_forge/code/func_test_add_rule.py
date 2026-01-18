from tests.compat import unittest
from tests.unit import AWSMockServiceTestCase
from boto.ec2.connection import EC2Connection
from boto.ec2.securitygroup import SecurityGroup
def test_add_rule(self):
    sg = SecurityGroup()
    self.assertEqual(len(sg.rules), 0)
    sg.add_rule(ip_protocol='http', from_port='80', to_port='8080', src_group_name='groupy', src_group_owner_id='12345', cidr_ip='10.0.0.1', src_group_group_id='54321', dry_run=False)
    self.assertEqual(len(sg.rules), 1)