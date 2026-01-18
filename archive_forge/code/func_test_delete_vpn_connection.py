from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.vpc import VPCConnection, VpnConnection
def test_delete_vpn_connection(self):
    self.set_http_response(status_code=200)
    api_response = self.service_connection.delete_vpn_connection('vpn-44a8938f')
    self.assert_request_parameters({'Action': 'DeleteVpnConnection', 'VpnConnectionId': 'vpn-44a8938f'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])
    self.assertEquals(api_response, True)