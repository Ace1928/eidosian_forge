from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.vpc import VPCConnection, RouteTable
def test_delete_route_table(self):
    self.set_http_response(status_code=200)
    api_response = self.service_connection.delete_route_table('rtb-e4ad488d')
    self.assert_request_parameters({'Action': 'DeleteRouteTable', 'RouteTableId': 'rtb-e4ad488d'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])
    self.assertEquals(api_response, True)