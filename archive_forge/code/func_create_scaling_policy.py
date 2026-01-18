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
def create_scaling_policy(self, scaling_policy):
    """
        Creates a new Scaling Policy.

        :type scaling_policy: :class:`boto.ec2.autoscale.policy.ScalingPolicy`
        :param scaling_policy: ScalingPolicy object.
        """
    params = {'AdjustmentType': scaling_policy.adjustment_type, 'AutoScalingGroupName': scaling_policy.as_name, 'PolicyName': scaling_policy.name, 'ScalingAdjustment': scaling_policy.scaling_adjustment}
    if scaling_policy.adjustment_type == 'PercentChangeInCapacity' and scaling_policy.min_adjustment_step is not None:
        params['MinAdjustmentStep'] = scaling_policy.min_adjustment_step
    if scaling_policy.cooldown is not None:
        params['Cooldown'] = scaling_policy.cooldown
    return self.get_object('PutScalingPolicy', params, Request)