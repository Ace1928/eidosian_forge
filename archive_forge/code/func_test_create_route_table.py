from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.vpc import VPCConnection, RouteTable
def test_create_route_table(self):
    self.set_http_response(status_code=200)
    api_response = self.service_connection.create_route_table('vpc-11ad4878')
    self.assert_request_parameters({'Action': 'CreateRouteTable', 'VpcId': 'vpc-11ad4878'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])
    self.assertIsInstance(api_response, RouteTable)
    self.assertEquals(api_response.id, 'rtb-f9ad4890')
    self.assertEquals(len(api_response.routes), 1)
    self.assertEquals(api_response.routes[0].destination_cidr_block, '10.0.0.0/22')
    self.assertEquals(api_response.routes[0].gateway_id, 'local')
    self.assertEquals(api_response.routes[0].state, 'active')