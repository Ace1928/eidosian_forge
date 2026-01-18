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
def bundle_instance(self, instance_id, s3_bucket, s3_prefix, s3_upload_policy, dry_run=False):
    """
        Bundle Windows instance.

        :type instance_id: string
        :param instance_id: The instance id

        :type s3_bucket: string
        :param s3_bucket: The bucket in which the AMI should be stored.

        :type s3_prefix: string
        :param s3_prefix: The beginning of the file name for the AMI.

        :type s3_upload_policy: string
        :param s3_upload_policy: Base64 encoded policy that specifies condition
                                 and permissions for Amazon EC2 to upload the
                                 user's image into Amazon S3.

        :type dry_run: bool
        :param dry_run: Set to True if the operation should not actually run.

        """
    params = {'InstanceId': instance_id, 'Storage.S3.Bucket': s3_bucket, 'Storage.S3.Prefix': s3_prefix, 'Storage.S3.UploadPolicy': s3_upload_policy}
    s3auth = boto.auth.get_auth_handler(None, boto.config, self.provider, ['s3'])
    params['Storage.S3.AWSAccessKeyId'] = self.aws_access_key_id
    signature = s3auth.sign_string(s3_upload_policy)
    params['Storage.S3.UploadPolicySignature'] = signature
    if dry_run:
        params['DryRun'] = 'true'
    return self.get_object('BundleInstance', params, BundleInstanceTask, verb='POST')