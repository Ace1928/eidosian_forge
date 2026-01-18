from tests.compat import OrderedDict
from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.vpc import VPCConnection, CustomerGateway
def test_delete_customer_gateway(self):
    self.set_http_response(status_code=200)
    api_response = self.service_connection.delete_customer_gateway('cgw-b4dc3961')
    self.assert_request_parameters({'Action': 'DeleteCustomerGateway', 'CustomerGatewayId': 'cgw-b4dc3961'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])
    self.assertEquals(api_response, True)