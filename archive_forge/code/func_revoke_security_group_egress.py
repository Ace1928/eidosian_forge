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
def revoke_security_group_egress(self, group_id, ip_protocol, from_port=None, to_port=None, src_group_id=None, cidr_ip=None, dry_run=False):
    """
        Remove an existing egress rule from an existing VPC security
        group.  You need to pass in an ip_protocol, from_port and
        to_port range only if the protocol you are using is
        port-based. You also need to pass in either a src_group_id or
        cidr_ip.

        :type group_name: string
        :param group_id:  The name of the security group you are removing
            the rule from.

        :type ip_protocol: string
        :param ip_protocol: Either tcp | udp | icmp | -1

        :type from_port: int
        :param from_port: The beginning port number you are disabling

        :type to_port: int
        :param to_port: The ending port number you are disabling

        :type src_group_id: src_group_id
        :param src_group_id: The source security group you are
            revoking access to.

        :type cidr_ip: string
        :param cidr_ip: The CIDR block you are revoking access to.
            See http://goo.gl/Yj5QC

        :type dry_run: bool
        :param dry_run: Set to True if the operation should not actually run.

        :rtype: bool
        :return: True if successful.
        """
    params = {}
    if group_id:
        params['GroupId'] = group_id
    if ip_protocol:
        params['IpPermissions.1.IpProtocol'] = ip_protocol
    if from_port is not None:
        params['IpPermissions.1.FromPort'] = from_port
    if to_port is not None:
        params['IpPermissions.1.ToPort'] = to_port
    if src_group_id is not None:
        params['IpPermissions.1.Groups.1.GroupId'] = src_group_id
    if cidr_ip:
        params['IpPermissions.1.IpRanges.1.CidrIp'] = cidr_ip
    if dry_run:
        params['DryRun'] = 'true'
    return self.get_status('RevokeSecurityGroupEgress', params, verb='POST')