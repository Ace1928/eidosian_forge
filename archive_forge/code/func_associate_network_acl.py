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
def associate_network_acl(self, network_acl_id, subnet_id):
    """
        Associates a network acl with a specific subnet.

        :type network_acl_id: str
        :param network_acl_id: The ID of the network ACL to associate.

        :type subnet_id: str
        :param subnet_id: The ID of the subnet to associate with.

        :rtype: str
        :return: The ID of the association created
        """
    acl = self.get_all_network_acls(filters=[('association.subnet-id', subnet_id)])[0]
    association = [association for association in acl.associations if association.subnet_id == subnet_id][0]
    params = {'AssociationId': association.id, 'NetworkAclId': network_acl_id}
    result = self.get_object('ReplaceNetworkAclAssociation', params, ResultSet)
    return result.newAssociationId