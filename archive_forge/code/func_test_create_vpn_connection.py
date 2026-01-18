from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.vpc import VPCConnection, VpnConnection
def test_create_vpn_connection(self):
    self.set_http_response(status_code=200)
    api_response = self.service_connection.create_vpn_connection('ipsec.1', 'cgw-b4dc3961', 'vgw-8db04f81', static_routes_only=True)
    self.assert_request_parameters({'Action': 'CreateVpnConnection', 'Type': 'ipsec.1', 'CustomerGatewayId': 'cgw-b4dc3961', 'VpnGatewayId': 'vgw-8db04f81', 'Options.StaticRoutesOnly': 'true'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])
    self.assertIsInstance(api_response, VpnConnection)
    self.assertEquals(api_response.id, 'vpn-83ad48ea')
    self.assertEquals(api_response.customer_gateway_id, 'cgw-b4dc3961')
    self.assertEquals(api_response.options.static_routes_only, True)