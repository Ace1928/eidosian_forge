from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.vpc import VPCConnection, InternetGateway
def test_attach_internet_gateway(self):
    self.set_http_response(status_code=200)
    api_response = self.service_connection.attach_internet_gateway('igw-eaad4883', 'vpc-11ad4878')
    self.assert_request_parameters({'Action': 'AttachInternetGateway', 'InternetGatewayId': 'igw-eaad4883', 'VpcId': 'vpc-11ad4878'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])
    self.assertEquals(api_response, True)