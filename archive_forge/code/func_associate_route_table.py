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
def associate_route_table(self, route_table_id, subnet_id, dry_run=False):
    """
        Associates a route table with a specific subnet.

        :type route_table_id: str
        :param route_table_id: The ID of the route table to associate.

        :type subnet_id: str
        :param subnet_id: The ID of the subnet to associate with.

        :type dry_run: bool
        :param dry_run: Set to True if the operation should not actually run.

        :rtype: str
        :return: The ID of the association created
        """
    params = {'RouteTableId': route_table_id, 'SubnetId': subnet_id}
    if dry_run:
        params['DryRun'] = 'true'
    result = self.get_object('AssociateRouteTable', params, ResultSet)
    return result.associationId