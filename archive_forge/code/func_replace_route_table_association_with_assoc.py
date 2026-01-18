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
def replace_route_table_association_with_assoc(self, association_id, route_table_id, dry_run=False):
    """
        Replaces a route association with a new route table.  This can be
        used to replace the 'main' route table by using the main route
        table association instead of the more common subnet type
        association. Returns the new association ID.

        :type association_id: str
        :param association_id: The ID of the existing association to replace.

        :type route_table_id: str
        :param route_table_id: The route table to ID to be used in the
            association.

        :type dry_run: bool
        :param dry_run: Set to True if the operation should not actually run.

        :rtype: str
        :return: New association ID
        """
    return self._replace_route_table_association(association_id, route_table_id, dry_run=dry_run).newAssociationId