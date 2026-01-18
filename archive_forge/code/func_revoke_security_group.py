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
def revoke_security_group(self, group_name=None, src_security_group_name=None, src_security_group_owner_id=None, ip_protocol=None, from_port=None, to_port=None, cidr_ip=None, group_id=None, src_security_group_group_id=None, dry_run=False):
    """
        Remove an existing rule from an existing security group.
        You need to pass in either src_security_group_name and
        src_security_group_owner_id OR ip_protocol, from_port, to_port,
        and cidr_ip.  In other words, either you are revoking another
        group or you are revoking some ip-based rule.

        :type group_name: string
        :param group_name: The name of the security group you are removing
            the rule from.

        :type src_security_group_name: string
        :param src_security_group_name: The name of the security group you are
            revoking access to.

        :type src_security_group_owner_id: string
        :param src_security_group_owner_id: The ID of the owner of the security
            group you are revoking access to.

        :type ip_protocol: string
        :param ip_protocol: Either tcp | udp | icmp

        :type from_port: int
        :param from_port: The beginning port number you are disabling

        :type to_port: int
        :param to_port: The ending port number you are disabling

        :type cidr_ip: string
        :param cidr_ip: The CIDR block you are revoking access to.
            See http://goo.gl/Yj5QC

        :type group_id: string
        :param group_id: ID of the EC2 or VPC security group to
            modify.  This is required for VPC security groups and can
            be used instead of group_name for EC2 security groups.

        :type src_security_group_group_id: string
        :param src_security_group_group_id: The ID of the security group
            for which you are revoking access.  Can be used instead
            of src_security_group_name

        :type dry_run: bool
        :param dry_run: Set to True if the operation should not actually run.

        :rtype: bool
        :return: True if successful.
        """
    if src_security_group_name:
        if from_port is None and to_port is None and (ip_protocol is None):
            return self.revoke_security_group_deprecated(group_name, src_security_group_name, src_security_group_owner_id)
    params = {}
    if group_name is not None:
        params['GroupName'] = group_name
    if group_id is not None:
        params['GroupId'] = group_id
    if src_security_group_name:
        param_name = 'IpPermissions.1.Groups.1.GroupName'
        params[param_name] = src_security_group_name
    if src_security_group_group_id:
        param_name = 'IpPermissions.1.Groups.1.GroupId'
        params[param_name] = src_security_group_group_id
    if src_security_group_owner_id:
        param_name = 'IpPermissions.1.Groups.1.UserId'
        params[param_name] = src_security_group_owner_id
    if ip_protocol:
        params['IpPermissions.1.IpProtocol'] = ip_protocol
    if from_port is not None:
        params['IpPermissions.1.FromPort'] = from_port
    if to_port is not None:
        params['IpPermissions.1.ToPort'] = to_port
    if cidr_ip:
        params['IpPermissions.1.IpRanges.1.CidrIp'] = cidr_ip
    if dry_run:
        params['DryRun'] = 'true'
    return self.get_status('RevokeSecurityGroupIngress', params, verb='POST')