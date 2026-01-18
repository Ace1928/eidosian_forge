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
def get_all_spot_instance_requests(self, request_ids=None, filters=None, dry_run=False):
    """
        Retrieve all the spot instances requests associated with your account.

        :type request_ids: list
        :param request_ids: A list of strings of spot instance request IDs

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
        :return: A list of
                 :class:`boto.ec2.spotinstancerequest.SpotInstanceRequest`
        """
    params = {}
    if request_ids:
        self.build_list_params(params, request_ids, 'SpotInstanceRequestId')
    if filters:
        if 'launch.group-id' in filters:
            lgid = filters.get('launch.group-id')
            if not lgid.startswith('sg-') or len(lgid) != 11:
                warnings.warn("The 'launch.group-id' filter now requires a security group id (sg-*) and no longer supports filtering by group name. Please update your filters accordingly.", UserWarning)
        self.build_filter_params(params, filters)
    if dry_run:
        params['DryRun'] = 'true'
    return self.get_list('DescribeSpotInstanceRequests', params, [('item', SpotInstanceRequest)], verb='POST')