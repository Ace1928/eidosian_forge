from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.vpc import VPCConnection
def test_get_network_acl_associations(self):
    self.set_http_response(status_code=200)
    api_response = self.service_connection.get_all_network_acls()
    association = api_response[0].associations[0]
    self.assertEqual(association.network_acl_id, 'acl-5d659634')