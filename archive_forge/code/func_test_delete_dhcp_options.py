from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.vpc import VPCConnection, DhcpOptions
def test_delete_dhcp_options(self):
    self.set_http_response(status_code=200)
    api_response = self.service_connection.delete_dhcp_options('dopt-7a8b9c2d')
    self.assert_request_parameters({'Action': 'DeleteDhcpOptions', 'DhcpOptionsId': 'dopt-7a8b9c2d'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])
    self.assertEquals(api_response, True)