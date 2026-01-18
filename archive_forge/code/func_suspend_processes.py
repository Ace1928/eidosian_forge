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
def suspend_processes(self, as_group, scaling_processes=None):
    """
        Suspends Auto Scaling processes for an Auto Scaling group.

        :type as_group: string
        :param as_group: The auto scaling group to suspend processes on.

        :type scaling_processes: list
        :param scaling_processes: Processes you want to suspend. If omitted,
            all processes will be suspended.
        """
    params = {'AutoScalingGroupName': as_group}
    if scaling_processes:
        self.build_list_params(params, scaling_processes, 'ScalingProcesses')
    return self.get_status('SuspendProcesses', params)