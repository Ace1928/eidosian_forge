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
def execute_policy(self, policy_name, as_group=None, honor_cooldown=None):
    params = {'PolicyName': policy_name}
    if as_group:
        params['AutoScalingGroupName'] = as_group
    if honor_cooldown:
        params['HonorCooldown'] = honor_cooldown
    return self.get_status('ExecutePolicy', params)