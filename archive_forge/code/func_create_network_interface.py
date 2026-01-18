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
def create_network_interface(self, subnet_id, private_ip_address=None, description=None, groups=None, dry_run=False):
    """
        Creates a network interface in the specified subnet.

        :type subnet_id: str
        :param subnet_id: The ID of the subnet to associate with the
            network interface.

        :type private_ip_address: str
        :param private_ip_address: The private IP address of the
            network interface.  If not supplied, one will be chosen
            for you.

        :type description: str
        :param description: The description of the network interface.

        :type groups: list
        :param groups: Lists the groups for use by the network interface.
            This can be either a list of group ID's or a list of
            :class:`boto.ec2.securitygroup.SecurityGroup` objects.

        :type dry_run: bool
        :param dry_run: Set to True if the operation should not actually run.

        :rtype: :class:`boto.ec2.networkinterface.NetworkInterface`
        :return: The newly created network interface.
        """
    params = {'SubnetId': subnet_id}
    if private_ip_address:
        params['PrivateIpAddress'] = private_ip_address
    if description:
        params['Description'] = description
    if groups:
        ids = []
        for group in groups:
            if isinstance(group, SecurityGroup):
                ids.append(group.id)
            else:
                ids.append(group)
        self.build_list_params(params, ids, 'SecurityGroupId')
    if dry_run:
        params['DryRun'] = 'true'
    return self.get_object('CreateNetworkInterface', params, NetworkInterface, verb='POST')