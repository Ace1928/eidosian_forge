from tests.compat import OrderedDict
from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.vpc import VPCConnection, Subnet
def test_delete_subnet(self):
    self.set_http_response(status_code=200)
    api_response = self.service_connection.delete_subnet('subnet-9d4a7b6c')
    self.assert_request_parameters({'Action': 'DeleteSubnet', 'SubnetId': 'subnet-9d4a7b6c'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])
    self.assertEquals(api_response, True)