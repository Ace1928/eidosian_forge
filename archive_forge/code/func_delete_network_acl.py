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
def delete_network_acl(self, network_acl_id):
    """
        Delete a network ACL

        :type network_acl_id: str
        :param network_acl_id: The ID of the network_acl to delete.

        :rtype: bool
        :return: True if successful
        """
    params = {'NetworkAclId': network_acl_id}
    return self.get_status('DeleteNetworkAcl', params)