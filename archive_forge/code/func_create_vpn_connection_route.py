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
def create_vpn_connection_route(self, destination_cidr_block, vpn_connection_id, dry_run=False):
    """
        Creates a new static route associated with a VPN connection between an
        existing virtual private gateway and a VPN customer gateway. The static
        route allows traffic to be routed from the virtual private gateway to
        the VPN customer gateway.

        :type destination_cidr_block: str
        :param destination_cidr_block: The CIDR block associated with the local
            subnet of the customer data center.

        :type vpn_connection_id: str
        :param vpn_connection_id: The ID of the VPN connection.

        :type dry_run: bool
        :param dry_run: Set to True if the operation should not actually run.

        :rtype: bool
        :return: True if successful
        """
    params = {'DestinationCidrBlock': destination_cidr_block, 'VpnConnectionId': vpn_connection_id}
    if dry_run:
        params['DryRun'] = 'true'
    return self.get_status('CreateVpnConnectionRoute', params)