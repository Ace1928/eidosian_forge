from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.vpc import VPCConnection, InternetGateway
def test_create_internet_gateway(self):
    self.set_http_response(status_code=200)
    api_response = self.service_connection.create_internet_gateway()
    self.assert_request_parameters({'Action': 'CreateInternetGateway'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])
    self.assertIsInstance(api_response, InternetGateway)
    self.assertEqual(api_response.id, 'igw-eaad4883')