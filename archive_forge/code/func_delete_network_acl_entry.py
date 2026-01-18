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
def delete_network_acl_entry(self, network_acl_id, rule_number, egress=None):
    """
        Deletes a network ACL entry from a network ACL within a VPC.

        :type network_acl_id: str
        :param network_acl_id: The ID of the network ACL with the network ACL entry.

        :type rule_number: int
        :param rule_number: The rule number for the entry to delete.

        :type egress: bool
        :param egress: Specifies whether the rule to delete is an egress rule (true)
        or ingress rule (false).

        :rtype: bool
        :return: True if successful
        """
    params = {'NetworkAclId': network_acl_id, 'RuleNumber': rule_number}
    if egress is not None:
        if isinstance(egress, bool):
            egress = str(egress).lower()
        params['Egress'] = egress
    return self.get_status('DeleteNetworkAclEntry', params)