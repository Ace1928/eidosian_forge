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
def reboot_instances(self, instance_ids=None, dry_run=False):
    """
        Reboot the specified instances.

        :type instance_ids: list
        :param instance_ids: The instances to terminate and reboot

        :type dry_run: bool
        :param dry_run: Set to True if the operation should not actually run.

        """
    params = {}
    if instance_ids:
        self.build_list_params(params, instance_ids, 'InstanceId')
    if dry_run:
        params['DryRun'] = 'true'
    return self.get_status('RebootInstances', params)