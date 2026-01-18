from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.vpc import VPCConnection, VPC
from boto.ec2.securitygroup import SecurityGroup
def test_create_vpc(self):
    self.set_http_response(status_code=200)
    api_response = self.service_connection.create_vpc('10.0.0.0/16', 'default')
    self.assert_request_parameters({'Action': 'CreateVpc', 'InstanceTenancy': 'default', 'CidrBlock': '10.0.0.0/16'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])
    self.assertIsInstance(api_response, VPC)
    self.assertEquals(api_response.id, 'vpc-1a2b3c4d')
    self.assertEquals(api_response.state, 'pending')
    self.assertEquals(api_response.cidr_block, '10.0.0.0/16')
    self.assertEquals(api_response.dhcp_options_id, 'dopt-1a2b3c4d2')
    self.assertEquals(api_response.instance_tenancy, 'default')