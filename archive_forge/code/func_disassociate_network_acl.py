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
def disassociate_network_acl(self, subnet_id, vpc_id=None):
    """
        Figures out what the default ACL is for the VPC, and associates
        current network ACL with the default.

        :type subnet_id: str
        :param subnet_id: The ID of the subnet to which the ACL belongs.

        :type vpc_id: str
        :param vpc_id: The ID of the VPC to which the ACL/subnet belongs. Queries EC2 if omitted.

        :rtype: str
        :return: The ID of the association created
        """
    if not vpc_id:
        vpc_id = self.get_all_subnets([subnet_id])[0].vpc_id
    acls = self.get_all_network_acls(filters=[('vpc-id', vpc_id), ('default', 'true')])
    default_acl_id = acls[0].id
    return self.associate_network_acl(default_acl_id, subnet_id)