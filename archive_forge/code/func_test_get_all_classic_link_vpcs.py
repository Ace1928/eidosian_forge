from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.vpc import VPCConnection, VPC
from boto.ec2.securitygroup import SecurityGroup
def test_get_all_classic_link_vpcs(self):
    self.set_http_response(status_code=200)
    response = self.service_connection.get_all_classic_link_vpcs()
    self.assertEqual(len(response), 2)
    vpc = response[0]
    self.assertEqual(vpc.id, 'vpc-6226ab07')
    self.assertEqual(vpc.classic_link_enabled, 'false')
    self.assertEqual(vpc.tags, {'Name': 'hello'})