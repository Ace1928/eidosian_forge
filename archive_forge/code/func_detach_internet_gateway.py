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
def detach_internet_gateway(self, internet_gateway_id, vpc_id, dry_run=False):
    """
        Detach an internet gateway from a specific VPC.

        :type internet_gateway_id: str
        :param internet_gateway_id: The ID of the internet gateway to detach.

        :type vpc_id: str
        :param vpc_id: The ID of the VPC to attach to.

        :type dry_run: bool
        :param dry_run: Set to True if the operation should not actually run.

        :rtype: Bool
        :return: True if successful
        """
    params = {'InternetGatewayId': internet_gateway_id, 'VpcId': vpc_id}
    if dry_run:
        params['DryRun'] = 'true'
    return self.get_status('DetachInternetGateway', params)