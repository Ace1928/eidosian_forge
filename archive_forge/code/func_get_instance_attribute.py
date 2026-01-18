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
def get_instance_attribute(self, instance_id, attribute, dry_run=False):
    """
        Gets an attribute from an instance.

        :type instance_id: string
        :param instance_id: The Amazon id of the instance

        :type attribute: string
        :param attribute: The attribute you need information about
            Valid choices are:

            * instanceType
            * kernel
            * ramdisk
            * userData
            * disableApiTermination
            * instanceInitiatedShutdownBehavior
            * rootDeviceName
            * blockDeviceMapping
            * productCodes
            * sourceDestCheck
            * groupSet
            * ebsOptimized
            * sriovNetSupport

        :type dry_run: bool
        :param dry_run: Set to True if the operation should not actually run.

        :rtype: :class:`boto.ec2.image.InstanceAttribute`
        :return: An InstanceAttribute object representing the value of the
                 attribute requested
        """
    params = {'InstanceId': instance_id}
    if attribute:
        params['Attribute'] = attribute
    if dry_run:
        params['DryRun'] = 'true'
    return self.get_object('DescribeInstanceAttribute', params, InstanceAttribute, verb='POST')