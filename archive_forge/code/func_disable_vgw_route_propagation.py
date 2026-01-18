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
def disable_vgw_route_propagation(self, route_table_id, gateway_id, dry_run=False):
    """
        Disables a virtual private gateway (VGW) from propagating routes to the
        routing tables of an Amazon VPC.

        :type route_table_id: str
        :param route_table_id: The ID of the routing table.

        :type gateway_id: str
        :param gateway_id: The ID of the virtual private gateway.

        :type dry_run: bool
        :param dry_run: Set to True if the operation should not actually run.

        :rtype: bool
        :return: True if successful
        """
    params = {'RouteTableId': route_table_id, 'GatewayId': gateway_id}
    if dry_run:
        params['DryRun'] = 'true'
    return self.get_status('DisableVgwRoutePropagation', params)