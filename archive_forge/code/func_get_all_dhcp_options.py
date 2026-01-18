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
def get_all_dhcp_options(self, dhcp_options_ids=None, filters=None, dry_run=False):
    """
        Retrieve information about your DhcpOptions.

        :type dhcp_options_ids: list
        :param dhcp_options_ids: A list of strings with the desired DhcpOption ID's

        :type filters: list of tuples or dict
        :param filters: A list of tuples or dict containing filters.  Each tuple
            or dict item consists of a filter key and a filter value.

        :type dry_run: bool
        :param dry_run: Set to True if the operation should not actually run.

        :rtype: list
        :return: A list of :class:`boto.vpc.dhcpoptions.DhcpOptions`
        """
    params = {}
    if dhcp_options_ids:
        self.build_list_params(params, dhcp_options_ids, 'DhcpOptionsId')
    if filters:
        self.build_filter_params(params, filters)
    if dry_run:
        params['DryRun'] = 'true'
    return self.get_list('DescribeDhcpOptions', params, [('item', DhcpOptions)])