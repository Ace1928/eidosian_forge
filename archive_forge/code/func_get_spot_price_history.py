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
def get_spot_price_history(self, start_time=None, end_time=None, instance_type=None, product_description=None, availability_zone=None, dry_run=False, max_results=None, next_token=None, filters=None):
    """
        Retrieve the recent history of spot instances pricing.

        :type start_time: str
        :param start_time: An indication of how far back to provide price
            changes for. An ISO8601 DateTime string.

        :type end_time: str
        :param end_time: An indication of how far forward to provide price
            changes for.  An ISO8601 DateTime string.

        :type instance_type: str
        :param instance_type: Filter responses to a particular instance type.

        :type product_description: str
        :param product_description: Filter responses to a particular platform.
            Valid values are currently:

            * Linux/UNIX
            * SUSE Linux
            * Windows
            * Linux/UNIX (Amazon VPC)
            * SUSE Linux (Amazon VPC)
            * Windows (Amazon VPC)

        :type availability_zone: str
        :param availability_zone: The availability zone for which prices
            should be returned.  If not specified, data for all
            availability zones will be returned.

        :type dry_run: bool
        :param dry_run: Set to True if the operation should not actually run.

        :type max_results: int
        :param max_results: The maximum number of paginated items
            per response.

        :type next_token: str
        :param next_token: The next set of rows to return.  This should
            be the value of the ``next_token`` attribute from a previous
            call to ``get_spot_price_history``.

        :type filters: dict
        :param filters: Optional filters that can be used to limit the
            results returned.  Filters are provided in the form of a
            dictionary consisting of filter names as the key and
            filter values as the value.  The set of allowable filter
            names/values is dependent on the request being performed.
            Check the EC2 API guide for details.

        :rtype: list
        :return: A list tuples containing price and timestamp.
        """
    params = {}
    if start_time:
        params['StartTime'] = start_time
    if end_time:
        params['EndTime'] = end_time
    if instance_type:
        params['InstanceType'] = instance_type
    if product_description:
        params['ProductDescription'] = product_description
    if availability_zone:
        params['AvailabilityZone'] = availability_zone
    if dry_run:
        params['DryRun'] = 'true'
    if max_results is not None:
        params['MaxResults'] = max_results
    if next_token:
        params['NextToken'] = next_token
    if filters:
        self.build_filter_params(params, filters)
    return self.get_list('DescribeSpotPriceHistory', params, [('item', SpotPriceHistory)], verb='POST')