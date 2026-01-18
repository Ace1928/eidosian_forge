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
def create_route_table(self, vpc_id, dry_run=False):
    """
        Creates a new route table.

        :type vpc_id: str
        :param vpc_id: The VPC ID to associate this route table with.

        :type dry_run: bool
        :param dry_run: Set to True if the operation should not actually run.

        :rtype: The newly created route table
        :return: A :class:`boto.vpc.routetable.RouteTable` object
        """
    params = {'VpcId': vpc_id}
    if dry_run:
        params['DryRun'] = 'true'
    return self.get_object('CreateRouteTable', params, RouteTable)