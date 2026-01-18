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
def get_all_reserved_instances_offerings(self, reserved_instances_offering_ids=None, instance_type=None, availability_zone=None, product_description=None, filters=None, instance_tenancy=None, offering_type=None, include_marketplace=None, min_duration=None, max_duration=None, max_instance_count=None, next_token=None, max_results=None, dry_run=False):
    """
        Describes Reserved Instance offerings that are available for purchase.

        :type reserved_instances_offering_ids: list
        :param reserved_instances_id: One or more Reserved Instances
            offering IDs.

        :type instance_type: str
        :param instance_type: Displays Reserved Instances of the specified
                              instance type.

        :type availability_zone: str
        :param availability_zone: Displays Reserved Instances within the
                                  specified Availability Zone.

        :type product_description: str
        :param product_description: Displays Reserved Instances with the
                                    specified product description.

        :type filters: dict
        :param filters: Optional filters that can be used to limit
                        the results returned.  Filters are provided
                        in the form of a dictionary consisting of
                        filter names as the key and filter values
                        as the value.  The set of allowable filter
                        names/values is dependent on the request
                        being performed.  Check the EC2 API guide
                        for details.

        :type instance_tenancy: string
        :param instance_tenancy: The tenancy of the Reserved Instance offering.
            A Reserved Instance with tenancy of dedicated will run on
            single-tenant hardware and can only be launched within a VPC.

        :type offering_type: string
        :param offering_type: The Reserved Instance offering type.  Valid
            Values: `"Heavy Utilization" | "Medium Utilization" | "Light
            Utilization"`

        :type include_marketplace: bool
        :param include_marketplace: Include Marketplace offerings in the
            response.

        :type min_duration: int :param min_duration: Minimum duration (in
            seconds) to filter when searching for offerings.

        :type max_duration: int
        :param max_duration: Maximum duration (in seconds) to filter when
            searching for offerings.

        :type max_instance_count: int
        :param max_instance_count: Maximum number of instances to filter when
            searching for offerings.

        :type next_token: string
        :param next_token: Token to use when requesting the next paginated set
            of offerings.

        :type max_results: int
        :param max_results: Maximum number of offerings to return per call.

        :type dry_run: bool
        :param dry_run: Set to True if the operation should not actually run.

        :rtype: list
        :return: A list of
            :class:`boto.ec2.reservedinstance.ReservedInstancesOffering`.

        """
    params = {}
    if reserved_instances_offering_ids is not None:
        self.build_list_params(params, reserved_instances_offering_ids, 'ReservedInstancesOfferingId')
    if instance_type:
        params['InstanceType'] = instance_type
    if availability_zone:
        params['AvailabilityZone'] = availability_zone
    if product_description:
        params['ProductDescription'] = product_description
    if filters:
        self.build_filter_params(params, filters)
    if instance_tenancy is not None:
        params['InstanceTenancy'] = instance_tenancy
    if offering_type is not None:
        params['OfferingType'] = offering_type
    if include_marketplace is not None:
        if include_marketplace:
            params['IncludeMarketplace'] = 'true'
        else:
            params['IncludeMarketplace'] = 'false'
    if min_duration is not None:
        params['MinDuration'] = str(min_duration)
    if max_duration is not None:
        params['MaxDuration'] = str(max_duration)
    if max_instance_count is not None:
        params['MaxInstanceCount'] = str(max_instance_count)
    if next_token is not None:
        params['NextToken'] = next_token
    if max_results is not None:
        params['MaxResults'] = str(max_results)
    if dry_run:
        params['DryRun'] = 'true'
    return self.get_list('DescribeReservedInstancesOfferings', params, [('item', ReservedInstancesOffering)], verb='POST')