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
def get_all_network_acls(self, network_acl_ids=None, filters=None):
    """
        Retrieve information about your network acls. You can filter results
        to return information only about those network acls that match your
        search parameters. Otherwise, all network acls associated with your
        account are returned.

        :type network_acl_ids: list
        :param network_acl_ids: A list of strings with the desired network ACL
                                IDs.

        :type filters: list of tuples or dict
        :param filters: A list of tuples or dict containing filters. Each tuple
                        or dict item consists of a filter key and a filter value.

        :rtype: list
        :return: A list of :class:`boto.vpc.networkacl.NetworkAcl`
        """
    params = {}
    if network_acl_ids:
        self.build_list_params(params, network_acl_ids, 'NetworkAclId')
    if filters:
        self.build_filter_params(params, filters)
    return self.get_list('DescribeNetworkAcls', params, [('item', NetworkAcl)])