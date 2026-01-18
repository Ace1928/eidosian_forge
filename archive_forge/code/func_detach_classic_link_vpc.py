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
def detach_classic_link_vpc(self, vpc_id, instance_id, dry_run=False):
    """
        Unlinks a linked EC2-Classic instance from a VPC. After the instance
        has been unlinked, the VPC security groups are no longer associated
        with it. An instance is automatically unlinked from a VPC when
        it's stopped.

        :type vpc_id: str
        :param vpc_id: The ID of the instance to unlink from the VPC.

        :type intance_id: str
        :param instance_is: The ID of the VPC to which the instance is linked.

        :type dry_run: bool
        :param dry_run: Set to True if the operation should not actually run.

        :rtype: bool
        :return: True if successful
        """
    params = {'VpcId': vpc_id, 'InstanceId': instance_id}
    if dry_run:
        params['DryRun'] = 'true'
    return self.get_status('DetachClassicLinkVpc', params)