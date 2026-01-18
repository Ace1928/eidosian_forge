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
def create_vpn_connection(self, type, customer_gateway_id, vpn_gateway_id, static_routes_only=None, dry_run=False):
    """
        Create a new VPN Connection.

        :type type: str
        :param type: The type of VPN Connection.  Currently only 'ipsec.1'
                     is supported

        :type customer_gateway_id: str
        :param customer_gateway_id: The ID of the customer gateway.

        :type vpn_gateway_id: str
        :param vpn_gateway_id: The ID of the VPN gateway.

        :type static_routes_only: bool
        :param static_routes_only: Indicates whether the VPN connection
        requires static routes. If you are creating a VPN connection
        for a device that does not support BGP, you must specify true.

        :type dry_run: bool
        :param dry_run: Set to True if the operation should not actually run.

        :rtype: The newly created VpnConnection
        :return: A :class:`boto.vpc.vpnconnection.VpnConnection` object
        """
    params = {'Type': type, 'CustomerGatewayId': customer_gateway_id, 'VpnGatewayId': vpn_gateway_id}
    if static_routes_only is not None:
        if isinstance(static_routes_only, bool):
            static_routes_only = str(static_routes_only).lower()
        params['Options.StaticRoutesOnly'] = static_routes_only
    if dry_run:
        params['DryRun'] = 'true'
    return self.get_object('CreateVpnConnection', params, VpnConnection)