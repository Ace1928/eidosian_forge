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
def insert_option(params, name, value):
    params['DhcpConfiguration.%d.Key' % (key_counter,)] = name
    if isinstance(value, (list, tuple)):
        for idx, value in enumerate(value, 1):
            key_name = 'DhcpConfiguration.%d.Value.%d' % (key_counter, idx)
            params[key_name] = value
    else:
        key_name = 'DhcpConfiguration.%d.Value.1' % (key_counter,)
        params[key_name] = value
    return key_counter + 1