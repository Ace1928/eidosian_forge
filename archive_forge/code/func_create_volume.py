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
def create_volume(self, size, zone, snapshot=None, volume_type=None, iops=None, encrypted=False, kms_key_id=None, dry_run=False):
    """
        Create a new EBS Volume.

        :type size: int
        :param size: The size of the new volume, in GiB

        :type zone: string or :class:`boto.ec2.zone.Zone`
        :param zone: The availability zone in which the Volume will be created.

        :type snapshot: string or :class:`boto.ec2.snapshot.Snapshot`
        :param snapshot: The snapshot from which the new Volume will be
            created.

        :type volume_type: string
        :param volume_type: The type of the volume. (optional).  Valid
            values are: standard | io1 | gp2.

        :type iops: int
        :param iops: The provisioned IOPS you want to associate with
            this volume. (optional)

        :type encrypted: bool
        :param encrypted: Specifies whether the volume should be encrypted.
            (optional)

        :type kms_key_id: string
        :params kms_key_id: If encrypted is True, this KMS Key ID may be specified to
            encrypt volume with this key (optional)
            e.g.: arn:aws:kms:us-east-1:012345678910:key/abcd1234-a123-456a-a12b-a123b4cd56ef

        :type dry_run: bool
        :param dry_run: Set to True if the operation should not actually run.

        """
    if isinstance(zone, Zone):
        zone = zone.name
    params = {'AvailabilityZone': zone}
    if size:
        params['Size'] = size
    if snapshot:
        if isinstance(snapshot, Snapshot):
            snapshot = snapshot.id
        params['SnapshotId'] = snapshot
    if volume_type:
        params['VolumeType'] = volume_type
    if iops:
        params['Iops'] = str(iops)
    if encrypted:
        params['Encrypted'] = 'true'
        if kms_key_id:
            params['KmsKeyId'] = kms_key_id
    if dry_run:
        params['DryRun'] = 'true'
    return self.get_object('CreateVolume', params, Volume, verb='POST')