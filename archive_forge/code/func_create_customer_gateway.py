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
def create_customer_gateway(self, type, ip_address, bgp_asn, dry_run=False):
    """
        Create a new Customer Gateway

        :type type: str
        :param type: Type of VPN Connection.  Only valid value currently is 'ipsec.1'

        :type ip_address: str
        :param ip_address: Internet-routable IP address for customer's gateway.
                           Must be a static address.

        :type bgp_asn: int
        :param bgp_asn: Customer gateway's Border Gateway Protocol (BGP)
                        Autonomous System Number (ASN)

        :type dry_run: bool
        :param dry_run: Set to True if the operation should not actually run.

        :rtype: The newly created CustomerGateway
        :return: A :class:`boto.vpc.customergateway.CustomerGateway` object
        """
    params = {'Type': type, 'IpAddress': ip_address, 'BgpAsn': bgp_asn}
    if dry_run:
        params['DryRun'] = 'true'
    return self.get_object('CreateCustomerGateway', params, CustomerGateway)