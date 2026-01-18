from tests.compat import OrderedDict
from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.vpc import VPCConnection, Subnet
def test_create_subnet(self):
    self.set_http_response(status_code=200)
    api_response = self.service_connection.create_subnet('vpc-1a2b3c4d', '10.0.1.0/24', 'us-east-1a')
    self.assert_request_parameters({'Action': 'CreateSubnet', 'VpcId': 'vpc-1a2b3c4d', 'CidrBlock': '10.0.1.0/24', 'AvailabilityZone': 'us-east-1a'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])
    self.assertIsInstance(api_response, Subnet)
    self.assertEquals(api_response.id, 'subnet-9d4a7b6c')
    self.assertEquals(api_response.state, 'pending')
    self.assertEquals(api_response.vpc_id, 'vpc-1a2b3c4d')
    self.assertEquals(api_response.cidr_block, '10.0.1.0/24')
    self.assertEquals(api_response.available_ip_address_count, 251)
    self.assertEquals(api_response.availability_zone, 'us-east-1a')