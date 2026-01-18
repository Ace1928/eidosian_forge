from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.vpc import VPCConnection, DhcpOptions
def test_get_all_dhcp_options(self):
    self.set_http_response(status_code=200)
    api_response = self.service_connection.get_all_dhcp_options(['dopt-7a8b9c2d'], [('key', 'domain-name')])
    self.assert_request_parameters({'Action': 'DescribeDhcpOptions', 'DhcpOptionsId.1': 'dopt-7a8b9c2d', 'Filter.1.Name': 'key', 'Filter.1.Value.1': 'domain-name'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])
    self.assertEquals(len(api_response), 1)
    self.assertIsInstance(api_response[0], DhcpOptions)
    self.assertEquals(api_response[0].id, 'dopt-7a8b9c2d')
    self.assertEquals(api_response[0].options['domain-name'], ['example.com'])
    self.assertEquals(api_response[0].options['domain-name-servers'], ['10.2.5.1', '10.2.5.2'])