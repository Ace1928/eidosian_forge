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
def get_all_images(self, image_ids=None, owners=None, executable_by=None, filters=None, dry_run=False):
    """
        Retrieve all the EC2 images available on your account.

        :type image_ids: list
        :param image_ids: A list of strings with the image IDs wanted

        :type owners: list
        :param owners: A list of owner IDs, the special strings 'self',
            'amazon', and 'aws-marketplace', may be used to describe
            images owned by you, Amazon or AWS Marketplace
            respectively

        :type executable_by: list
        :param executable_by: Returns AMIs for which the specified
                              user ID has explicit launch permissions

        :type filters: dict
        :param filters: Optional filters that can be used to limit the
            results returned.  Filters are provided in the form of a
            dictionary consisting of filter names as the key and
            filter values as the value.  The set of allowable filter
            names/values is dependent on the request being performed.
            Check the EC2 API guide for details.

        :type dry_run: bool
        :param dry_run: Set to True if the operation should not actually run.

        :rtype: list
        :return: A list of :class:`boto.ec2.image.Image`
        """
    params = {}
    if image_ids:
        self.build_list_params(params, image_ids, 'ImageId')
    if owners:
        self.build_list_params(params, owners, 'Owner')
    if executable_by:
        self.build_list_params(params, executable_by, 'ExecutableBy')
    if filters:
        self.build_filter_params(params, filters)
    if dry_run:
        params['DryRun'] = 'true'
    return self.get_list('DescribeImages', params, [('item', Image)], verb='POST')