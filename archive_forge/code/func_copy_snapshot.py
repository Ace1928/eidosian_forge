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
def copy_snapshot(self, source_region, source_snapshot_id, description=None, dry_run=False):
    """
        Copies a point-in-time snapshot of an Amazon Elastic Block Store
        (Amazon EBS) volume and stores it in Amazon Simple Storage Service
        (Amazon S3). You can copy the snapshot within the same region or from
        one region to another. You can use the snapshot to create new Amazon
        EBS volumes or Amazon Machine Images (AMIs).


        :type source_region: str
        :param source_region: The ID of the AWS region that contains the
            snapshot to be copied (e.g 'us-east-1', 'us-west-2', etc.).

        :type source_snapshot_id: str
        :param source_snapshot_id: The ID of the Amazon EBS snapshot to copy

        :type description: str
        :param description: A description of the new Amazon EBS snapshot.

        :type dry_run: bool
        :param dry_run: Set to True if the operation should not actually run.

        :rtype: str
        :return: The snapshot ID

        """
    params = {'SourceRegion': source_region, 'SourceSnapshotId': source_snapshot_id}
    if description is not None:
        params['Description'] = description
    if dry_run:
        params['DryRun'] = 'true'
    snapshot = self.get_object('CopySnapshot', params, Snapshot, verb='POST')
    return snapshot.id