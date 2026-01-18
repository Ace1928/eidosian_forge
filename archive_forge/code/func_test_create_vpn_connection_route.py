from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.vpc import VPCConnection, VpnConnection
def test_create_vpn_connection_route(self):
    self.set_http_response(status_code=200)
    api_response = self.service_connection.create_vpn_connection_route('11.12.0.0/16', 'vpn-83ad48ea')
    self.assert_request_parameters({'Action': 'CreateVpnConnectionRoute', 'DestinationCidrBlock': '11.12.0.0/16', 'VpnConnectionId': 'vpn-83ad48ea'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])
    self.assertEquals(api_response, True)