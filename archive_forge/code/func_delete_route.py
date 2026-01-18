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
def delete_route(self, route_table_id, destination_cidr_block, dry_run=False):
    """
        Deletes a route from a route table within a VPC.

        :type route_table_id: str
        :param route_table_id: The ID of the route table with the route.

        :type destination_cidr_block: str
        :param destination_cidr_block: The CIDR address block used for
                                       destination match.

        :type dry_run: bool
        :param dry_run: Set to True if the operation should not actually run.

        :rtype: bool
        :return: True if successful
        """
    params = {'RouteTableId': route_table_id, 'DestinationCidrBlock': destination_cidr_block}
    if dry_run:
        params['DryRun'] = 'true'
    return self.get_status('DeleteRoute', params)