from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.vpc import VPCConnection, VPC
from boto.ec2.securitygroup import SecurityGroup
def test_delete_vpc(self):
    self.set_http_response(status_code=200)
    api_response = self.service_connection.delete_vpc('vpc-1a2b3c4d')
    self.assert_request_parameters({'Action': 'DeleteVpc', 'VpcId': 'vpc-1a2b3c4d'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])
    self.assertEquals(api_response, True)