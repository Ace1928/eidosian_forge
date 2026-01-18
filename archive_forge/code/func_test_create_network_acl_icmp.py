from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.vpc import VPCConnection
def test_create_network_acl_icmp(self):
    self.set_http_response(status_code=200)
    response = self.service_connection.create_network_acl_entry('acl-2cb85d45', 110, 'udp', 'allow', '0.0.0.0/0', egress='true', icmp_code=-1, icmp_type=8)
    self.assert_request_parameters({'Action': 'CreateNetworkAclEntry', 'NetworkAclId': 'acl-2cb85d45', 'RuleNumber': 110, 'Protocol': 'udp', 'RuleAction': 'allow', 'Egress': 'true', 'CidrBlock': '0.0.0.0/0', 'Icmp.Code': -1, 'Icmp.Type': 8}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])
    self.assertEqual(response, True)