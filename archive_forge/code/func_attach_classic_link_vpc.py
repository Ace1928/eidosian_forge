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
def attach_classic_link_vpc(self, vpc_id, instance_id, groups, dry_run=False):
    """
        Links  an EC2-Classic instance to a ClassicLink-enabled VPC through one
        or more of the VPC's security groups. You cannot link an EC2-Classic
        instance to more than one VPC at a time. You can only link an instance
        that's in the running state. An instance is automatically unlinked from
        a VPC when it's stopped. You can link it to the VPC again when you
        restart it.

        After you've linked an instance, you cannot  change  the VPC security
        groups  that are associated with it. To change the security groups, you
        must first unlink the instance, and then link it again.

        Linking your instance to a VPC is sometimes referred  to  as  attaching
        your instance.

        :type vpc_id: str
        :param vpc_id: The ID of a ClassicLink-enabled VPC.

        :type intance_id: str
        :param instance_is: The ID of a ClassicLink-enabled VPC.

        :tye groups: list
        :param groups: The ID of one or more of the VPC's security groups.
            You cannot specify security groups from a different VPC. The
            members of the list can be
            :class:`boto.ec2.securitygroup.SecurityGroup` objects or
            strings of the id's of the security groups.

        :type dry_run: bool
        :param dry_run: Set to True if the operation should not actually run.

        :rtype: bool
        :return: True if successful
        """
    params = {'VpcId': vpc_id, 'InstanceId': instance_id}
    if dry_run:
        params['DryRun'] = 'true'
    l = []
    for group in groups:
        if hasattr(group, 'id'):
            l.append(group.id)
        else:
            l.append(group)
    self.build_list_params(params, l, 'SecurityGroupId')
    return self.get_status('AttachClassicLinkVpc', params)