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
def release_address(self, public_ip=None, allocation_id=None, dry_run=False):
    """
        Free up an Elastic IP address.  Pass a public IP address to
        release an EC2 Elastic IP address and an AllocationId to
        release a VPC Elastic IP address.  You should only pass
        one value.

        This requires one of ``public_ip`` or ``allocation_id`` depending
        on if you're associating a VPC address or a plain EC2 address.

        When using an Allocation ID, make sure to pass ``None`` for ``public_ip``
        as EC2 expects a single parameter and if ``public_ip`` is passed boto
        will preference that instead of ``allocation_id``.

        :type public_ip: string
        :param public_ip: The public IP address for EC2 elastic IPs.

        :type allocation_id: string
        :param allocation_id: The Allocation ID for VPC elastic IPs.

        :type dry_run: bool
        :param dry_run: Set to True if the operation should not actually run.

        :rtype: bool
        :return: True if successful
        """
    params = {}
    if public_ip is not None:
        params['PublicIp'] = public_ip
    elif allocation_id is not None:
        params['AllocationId'] = allocation_id
    if dry_run:
        params['DryRun'] = 'true'
    return self.get_status('ReleaseAddress', params, verb='POST')