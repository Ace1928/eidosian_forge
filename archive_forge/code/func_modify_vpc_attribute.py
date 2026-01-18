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
def modify_vpc_attribute(self, vpc_id, enable_dns_support=None, enable_dns_hostnames=None, dry_run=False):
    """
        Modifies the specified attribute of the specified VPC.
        You can only modify one attribute at a time.

        :type vpc_id: str
        :param vpc_id: The ID of the vpc to be deleted.

        :type enable_dns_support: bool
        :param enable_dns_support: Specifies whether the DNS server
            provided by Amazon is enabled for the VPC.

        :type enable_dns_hostnames: bool
        :param enable_dns_hostnames: Specifies whether DNS hostnames are
            provided for the instances launched in this VPC. You can only
            set this attribute to ``true`` if EnableDnsSupport
            is also ``true``.

        :type dry_run: bool
        :param dry_run: Set to True if the operation should not actually run.

        """
    params = {'VpcId': vpc_id}
    if enable_dns_support is not None:
        if enable_dns_support:
            params['EnableDnsSupport.Value'] = 'true'
        else:
            params['EnableDnsSupport.Value'] = 'false'
    if enable_dns_hostnames is not None:
        if enable_dns_hostnames:
            params['EnableDnsHostnames.Value'] = 'true'
        else:
            params['EnableDnsHostnames.Value'] = 'false'
    if dry_run:
        params['DryRun'] = 'true'
    return self.get_status('ModifyVpcAttribute', params)