import base64
import boto
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo, get_regions, load_regions
from boto.regioninfo import connect
from boto.ec2.autoscale.request import Request
from boto.ec2.autoscale.launchconfig import LaunchConfiguration
from boto.ec2.autoscale.group import AutoScalingGroup
from boto.ec2.autoscale.group import ProcessType
from boto.ec2.autoscale.activity import Activity
from boto.ec2.autoscale.policy import AdjustmentType
from boto.ec2.autoscale.policy import MetricCollectionTypes
from boto.ec2.autoscale.policy import ScalingPolicy
from boto.ec2.autoscale.policy import TerminationPolicies
from boto.ec2.autoscale.instance import Instance
from boto.ec2.autoscale.scheduled import ScheduledUpdateGroupAction
from boto.ec2.autoscale.tag import Tag
from boto.ec2.autoscale.limits import AccountLimits
from boto.compat import six
def create_launch_configuration(self, launch_config):
    """
        Creates a new Launch Configuration.

        :type launch_config: :class:`boto.ec2.autoscale.launchconfig.LaunchConfiguration`
        :param launch_config: LaunchConfiguration object.
        """
    params = {'ImageId': launch_config.image_id, 'LaunchConfigurationName': launch_config.name, 'InstanceType': launch_config.instance_type}
    if launch_config.key_name:
        params['KeyName'] = launch_config.key_name
    if launch_config.user_data:
        user_data = launch_config.user_data
        if isinstance(user_data, six.text_type):
            user_data = user_data.encode('utf-8')
        params['UserData'] = base64.b64encode(user_data).decode('utf-8')
    if launch_config.kernel_id:
        params['KernelId'] = launch_config.kernel_id
    if launch_config.ramdisk_id:
        params['RamdiskId'] = launch_config.ramdisk_id
    if launch_config.block_device_mappings:
        [x.autoscale_build_list_params(params) for x in launch_config.block_device_mappings]
    if launch_config.security_groups:
        self.build_list_params(params, launch_config.security_groups, 'SecurityGroups')
    if launch_config.instance_monitoring:
        params['InstanceMonitoring.Enabled'] = 'true'
    else:
        params['InstanceMonitoring.Enabled'] = 'false'
    if launch_config.spot_price is not None:
        params['SpotPrice'] = str(launch_config.spot_price)
    if launch_config.instance_profile_name is not None:
        params['IamInstanceProfile'] = launch_config.instance_profile_name
    if launch_config.ebs_optimized:
        params['EbsOptimized'] = 'true'
    else:
        params['EbsOptimized'] = 'false'
    if launch_config.associate_public_ip_address is True:
        params['AssociatePublicIpAddress'] = 'true'
    elif launch_config.associate_public_ip_address is False:
        params['AssociatePublicIpAddress'] = 'false'
    if launch_config.volume_type:
        params['VolumeType'] = launch_config.volume_type
    if launch_config.delete_on_termination:
        params['DeleteOnTermination'] = 'true'
    else:
        params['DeleteOnTermination'] = 'false'
    if launch_config.iops:
        params['Iops'] = launch_config.iops
    if launch_config.classic_link_vpc_id:
        params['ClassicLinkVPCId'] = launch_config.classic_link_vpc_id
    if launch_config.classic_link_vpc_security_groups:
        self.build_list_params(params, launch_config.classic_link_vpc_security_groups, 'ClassicLinkVPCSecurityGroups')
    return self.get_object('CreateLaunchConfiguration', params, Request, verb='POST')