from tests.compat import OrderedDict
from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.vpc import VPCConnection, CustomerGateway
def test_get_all_customer_gateways(self):
    self.set_http_response(status_code=200)
    api_response = self.service_connection.get_all_customer_gateways('cgw-b4dc3961', filters=OrderedDict([('state', ['pending', 'available']), ('ip-address', '12.1.2.3')]))
    self.assert_request_parameters({'Action': 'DescribeCustomerGateways', 'CustomerGatewayId.1': 'cgw-b4dc3961', 'Filter.1.Name': 'state', 'Filter.1.Value.1': 'pending', 'Filter.1.Value.2': 'available', 'Filter.2.Name': 'ip-address', 'Filter.2.Value.1': '12.1.2.3'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])
    self.assertEquals(len(api_response), 1)
    self.assertIsInstance(api_response[0], CustomerGateway)
    self.assertEqual(api_response[0].id, 'cgw-b4dc3961')