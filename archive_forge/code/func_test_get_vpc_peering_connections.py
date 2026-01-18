from tests.unit import mock, unittest
from tests.unit import AWSMockServiceTestCase
from boto.vpc import VpcPeeringConnection, VPCConnection, Subnet
def test_get_vpc_peering_connections(self):
    self.set_http_response(status_code=200)
    api_response = self.service_connection.get_all_vpc_peering_connections(['pcx-111aaa22', 'pcx-444bbb88'], filters=[('status-code', ['pending-acceptance'])])
    self.assertEqual(len(api_response), 2)
    for vpc_peering_connection in api_response:
        if vpc_peering_connection.id == 'pcx-111aaa22':
            self.assertEqual(vpc_peering_connection.id, 'pcx-111aaa22')
            self.assertEqual(vpc_peering_connection.status_code, 'pending-acceptance')
            self.assertEqual(vpc_peering_connection.status_message, 'Pending Acceptance by 111122223333')
            self.assertEqual(vpc_peering_connection.requester_vpc_info.owner_id, '777788889999')
            self.assertEqual(vpc_peering_connection.requester_vpc_info.vpc_id, 'vpc-1a2b3c4d')
            self.assertEqual(vpc_peering_connection.requester_vpc_info.cidr_block, '172.31.0.0/16')
            self.assertEqual(vpc_peering_connection.accepter_vpc_info.owner_id, '111122223333')
            self.assertEqual(vpc_peering_connection.accepter_vpc_info.vpc_id, 'vpc-aa22cc33')
            self.assertEqual(vpc_peering_connection.expiration_time, '2014-02-17T16:00:50.000Z')
        else:
            self.assertEqual(vpc_peering_connection.id, 'pcx-444bbb88')
            self.assertEqual(vpc_peering_connection.status_code, 'pending-acceptance')
            self.assertEqual(vpc_peering_connection.status_message, 'Pending Acceptance by 98654313')
            self.assertEqual(vpc_peering_connection.requester_vpc_info.owner_id, '1237897234')
            self.assertEqual(vpc_peering_connection.requester_vpc_info.vpc_id, 'vpc-2398abcd')
            self.assertEqual(vpc_peering_connection.requester_vpc_info.cidr_block, '172.30.0.0/16')
            self.assertEqual(vpc_peering_connection.accepter_vpc_info.owner_id, '98654313')
            self.assertEqual(vpc_peering_connection.accepter_vpc_info.vpc_id, 'vpc-0983bcda')
            self.assertEqual(vpc_peering_connection.expiration_time, '2015-02-17T16:00:50.000Z')