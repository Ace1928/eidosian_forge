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
def disable_metrics_collection(self, as_group, metrics=None):
    """
        Disables monitoring of group metrics for the Auto Scaling group
        specified in AutoScalingGroupName. You can specify the list of affected
        metrics with the Metrics parameter.
        """
    params = {'AutoScalingGroupName': as_group}
    if metrics:
        self.build_list_params(params, metrics, 'Metrics')
    return self.get_status('DisableMetricsCollection', params)