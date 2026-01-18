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
def modify_network_interface_attribute(self, interface_id, attr, value, attachment_id=None, dry_run=False):
    """
        Changes an attribute of a network interface.

        :type interface_id: string
        :param interface_id: The interface id. Looks like 'eni-xxxxxxxx'

        :type attr: string
        :param attr: The attribute you wish to change.

            Learn more at http://docs.aws.amazon.com/AWSEC2/latest/API            Reference/ApiReference-query-ModifyNetworkInterfaceAttribute.html

            * description - Textual description of interface
            * groupSet - List of security group ids or group objects
            * sourceDestCheck - Boolean
            * deleteOnTermination - Boolean. Must also specify attachment_id

        :type value: string
        :param value: The new value for the attribute

        :rtype: bool
        :return: Whether the operation succeeded or not

        :type attachment_id: string
        :param attachment_id: If you're modifying DeleteOnTermination you must
            specify the attachment_id.

        :type dry_run: bool
        :param dry_run: Set to True if the operation should not actually run.

        """
    bool_reqs = ('deleteontermination', 'sourcedestcheck')
    if attr.lower() in bool_reqs:
        if isinstance(value, bool):
            if value:
                value = 'true'
            else:
                value = 'false'
        elif value not in ['true', 'false']:
            raise ValueError('%s must be a boolean, "true", or "false"!' % attr)
    params = {'NetworkInterfaceId': interface_id}
    if attr.lower() == 'groupset':
        for idx, sg in enumerate(value):
            if isinstance(sg, SecurityGroup):
                sg = sg.id
            params['SecurityGroupId.%s' % (idx + 1)] = sg
    elif attr.lower() == 'description':
        params['Description.Value'] = value
    elif attr.lower() == 'sourcedestcheck':
        params['SourceDestCheck.Value'] = value
    elif attr.lower() == 'deleteontermination':
        params['Attachment.DeleteOnTermination'] = value
        if not attachment_id:
            raise ValueError('You must also specify an attachment_id')
        params['Attachment.AttachmentId'] = attachment_id
    else:
        raise ValueError('Unknown attribute "%s"' % (attr,))
    if dry_run:
        params['DryRun'] = 'true'
    return self.get_status('ModifyNetworkInterfaceAttribute', params, verb='POST')