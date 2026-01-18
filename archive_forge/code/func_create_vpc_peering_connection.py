from boto.ec2.connection import EC2Connection
from boto.resultset import ResultSet
from boto.vpc.vpc import VPC
from boto.vpc.customergateway import CustomerGateway
from boto.vpc.networkacl import NetworkAcl
from boto.vpc.routetable import RouteTable
from boto.vpc.internetgateway import InternetGateway
from boto.vpc.vpngateway import VpnGateway, Attachment
from boto.vpc.dhcpoptions import DhcpOptions
from boto.vpc.subnet import Subnet
from boto.vpc.vpnconnection import VpnConnection
from boto.vpc.vpc_peering_connection import VpcPeeringConnection
from boto.ec2 import RegionData
from boto.regioninfo import RegionInfo, get_regions
from boto.regioninfo import connect
def create_vpc_peering_connection(self, vpc_id, peer_vpc_id, peer_owner_id=None, dry_run=False):
    """
        Create a new VPN Peering connection.

        :type vpc_id: str
        :param vpc_id: The ID of the requester VPC.

        :type peer_vpc_id: str
        :param vpc_peer_id: The ID of the VPC with which you are creating the peering connection.

        :type peer_owner_id: str
        :param peer_owner_id: The AWS account ID of the owner of the peer VPC.

        :rtype: The newly created VpcPeeringConnection
        :return: A :class:`boto.vpc.vpc_peering_connection.VpcPeeringConnection` object
        """
    params = {'VpcId': vpc_id, 'PeerVpcId': peer_vpc_id}
    if peer_owner_id is not None:
        params['PeerOwnerId'] = peer_owner_id
    if dry_run:
        params['DryRun'] = 'true'
    return self.get_object('CreateVpcPeeringConnection', params, VpcPeeringConnection)