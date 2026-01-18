from tests.compat import OrderedDict
from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.vpc import VPCConnection, VpnGateway, Attachment
def test_delete_vpn_gateway(self):
    self.set_http_response(status_code=200)
    api_response = self.service_connection.delete_vpn_gateway('vgw-8db04f81')
    self.assert_request_parameters({'Action': 'DeleteVpnGateway', 'VpnGatewayId': 'vgw-8db04f81'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])
    self.assertEqual(api_response, True)