from tests.unit import mock, unittest
from tests.unit import AWSMockServiceTestCase
from boto.vpc import VpcPeeringConnection, VPCConnection, Subnet
def test_create_vpc_peering_connection(self):
    self.set_http_response(status_code=200)
    vpc_peering_connection = self.service_connection.create_vpc_peering_connection('vpc-1a2b3c4d', 'vpc-a1b2c3d4', '123456789012')
    self.assertEqual(vpc_peering_connection.id, 'pcx-73a5401a')
    self.assertEqual(vpc_peering_connection.status_code, 'initiating-request')
    self.assertEqual(vpc_peering_connection.status_message, 'Initiating Request to 123456789012')
    self.assertEqual(vpc_peering_connection.requester_vpc_info.owner_id, '777788889999')
    self.assertEqual(vpc_peering_connection.requester_vpc_info.vpc_id, 'vpc-1a2b3c4d')
    self.assertEqual(vpc_peering_connection.requester_vpc_info.cidr_block, '10.0.0.0/28')
    self.assertEqual(vpc_peering_connection.accepter_vpc_info.owner_id, '123456789012')
    self.assertEqual(vpc_peering_connection.accepter_vpc_info.vpc_id, 'vpc-a1b2c3d4')
    self.assertEqual(vpc_peering_connection.expiration_time, '2014-02-18T14:37:25.000Z')