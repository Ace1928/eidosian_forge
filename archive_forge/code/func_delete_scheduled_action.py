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
def delete_scheduled_action(self, scheduled_action_name, autoscale_group=None):
    """
        Deletes a previously scheduled action.

        :type scheduled_action_name: str
        :param scheduled_action_name: The name of the action you want
            to delete.

        :type autoscale_group: str
        :param autoscale_group: The name of the autoscale group.
        """
    params = {'ScheduledActionName': scheduled_action_name}
    if autoscale_group:
        params['AutoScalingGroupName'] = autoscale_group
    return self.get_status('DeleteScheduledAction', params)