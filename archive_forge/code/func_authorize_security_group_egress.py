import base64
import warnings
from datetime import datetime
from datetime import timedelta
import boto
from boto.auth import detect_potential_sigv4
from boto.connection import AWSQueryConnection
from boto.resultset import ResultSet
from boto.ec2.image import Image, ImageAttribute, CopyImage
from boto.ec2.instance import Reservation, Instance
from boto.ec2.instance import ConsoleOutput, InstanceAttribute
from boto.ec2.keypair import KeyPair
from boto.ec2.address import Address
from boto.ec2.volume import Volume, VolumeAttribute
from boto.ec2.snapshot import Snapshot
from boto.ec2.snapshot import SnapshotAttribute
from boto.ec2.zone import Zone
from boto.ec2.securitygroup import SecurityGroup
from boto.ec2.regioninfo import RegionInfo
from boto.ec2.instanceinfo import InstanceInfo
from boto.ec2.reservedinstance import ReservedInstancesOffering
from boto.ec2.reservedinstance import ReservedInstance
from boto.ec2.reservedinstance import ReservedInstanceListing
from boto.ec2.reservedinstance import ReservedInstancesConfiguration
from boto.ec2.reservedinstance import ModifyReservedInstancesResult
from boto.ec2.reservedinstance import ReservedInstancesModification
from boto.ec2.spotinstancerequest import SpotInstanceRequest
from boto.ec2.spotpricehistory import SpotPriceHistory
from boto.ec2.spotdatafeedsubscription import SpotDatafeedSubscription
from boto.ec2.bundleinstance import BundleInstanceTask
from boto.ec2.placementgroup import PlacementGroup
from boto.ec2.tag import Tag
from boto.ec2.instancetype import InstanceType
from boto.ec2.instancestatus import InstanceStatusSet
from boto.ec2.volumestatus import VolumeStatusSet
from boto.ec2.networkinterface import NetworkInterface
from boto.ec2.attributes import AccountAttribute, VPCAttribute
from boto.ec2.blockdevicemapping import BlockDeviceMapping, BlockDeviceType
from boto.exception import EC2ResponseError
from boto.compat import six
def authorize_security_group_egress(self, group_id, ip_protocol, from_port=None, to_port=None, src_group_id=None, cidr_ip=None, dry_run=False):
    """
        The action adds one or more egress rules to a VPC security
        group. Specifically, this action permits instances in a
        security group to send traffic to one or more destination
        CIDR IP address ranges, or to one or more destination
        security groups in the same VPC.

        :type dry_run: bool
        :param dry_run: Set to True if the operation should not actually run.

        """
    params = {'GroupId': group_id, 'IpPermissions.1.IpProtocol': ip_protocol}
    if from_port is not None:
        params['IpPermissions.1.FromPort'] = from_port
    if to_port is not None:
        params['IpPermissions.1.ToPort'] = to_port
    if src_group_id is not None:
        params['IpPermissions.1.Groups.1.GroupId'] = src_group_id
    if cidr_ip is not None:
        params['IpPermissions.1.IpRanges.1.CidrIp'] = cidr_ip
    if dry_run:
        params['DryRun'] = 'true'
    return self.get_status('AuthorizeSecurityGroupEgress', params, verb='POST')