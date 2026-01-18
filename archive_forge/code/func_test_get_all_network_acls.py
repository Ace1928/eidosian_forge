from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.vpc import VPCConnection
def test_get_all_network_acls(self):
    self.set_http_response(status_code=200)
    response = self.service_connection.get_all_network_acls(['acl-5566953c', 'acl-5d659634'], [('vpc-id', 'vpc-5266953b')])
    self.assert_request_parameters({'Action': 'DescribeNetworkAcls', 'NetworkAclId.1': 'acl-5566953c', 'NetworkAclId.2': 'acl-5d659634', 'Filter.1.Name': 'vpc-id', 'Filter.1.Value.1': 'vpc-5266953b'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])
    self.assertEqual(len(response), 2)