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
def set_instance_health(self, instance_id, health_status, should_respect_grace_period=True):
    """
        Explicitly set the health status of an instance.

        :type instance_id: str
        :param instance_id: The identifier of the EC2 instance.

        :type health_status: str
        :param health_status: The health status of the instance.
            "Healthy" means that the instance is healthy and should remain
            in service. "Unhealthy" means that the instance is unhealthy.
            Auto Scaling should terminate and replace it.

        :type should_respect_grace_period: bool
        :param should_respect_grace_period: If True, this call should
            respect the grace period associated with the group.
        """
    params = {'InstanceId': instance_id, 'HealthStatus': health_status}
    if should_respect_grace_period:
        params['ShouldRespectGracePeriod'] = 'true'
    else:
        params['ShouldRespectGracePeriod'] = 'false'
    return self.get_status('SetInstanceHealth', params)