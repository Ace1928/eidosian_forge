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
def get_all_vpcs(self, vpc_ids=None, filters=None, dry_run=False):
    """
        Retrieve information about your VPCs.  You can filter results to
        return information only about those VPCs that match your search
        parameters.  Otherwise, all VPCs associated with your account
        are returned.

        :type vpc_ids: list
        :param vpc_ids: A list of strings with the desired VPC ID's

        :type filters: list of tuples or dict
        :param filters: A list of tuples or dict containing filters.  Each tuple
            or dict item consists of a filter key and a filter value.
            Possible filter keys are:

            * *state* - a list of states of the VPC (pending or available)
            * *cidrBlock* - a list CIDR blocks of the VPC
            * *dhcpOptionsId* - a list of IDs of a set of DHCP options

        :type dry_run: bool
        :param dry_run: Set to True if the operation should not actually run.

        :rtype: list
        :return: A list of :class:`boto.vpc.vpc.VPC`
        """
    params = {}
    if vpc_ids:
        self.build_list_params(params, vpc_ids, 'VpcId')
    if filters:
        self.build_filter_params(params, filters)
    if dry_run:
        params['DryRun'] = 'true'
    return self.get_list('DescribeVpcs', params, [('item', VPC)])