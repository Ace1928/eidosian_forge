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
def get_all_activities(self, autoscale_group, activity_ids=None, max_records=None, next_token=None):
    """
        Get all activities for the given autoscaling group.

        This action supports pagination by returning a token if there are more
        pages to retrieve. To get the next page, call this action again with
        the returned token as the NextToken parameter

        :type autoscale_group: str or
            :class:`boto.ec2.autoscale.group.AutoScalingGroup` object
        :param autoscale_group: The auto scaling group to get activities on.

        :type max_records: int
        :param max_records: Maximum amount of activities to return.

        :rtype: list
        :returns: List of
            :class:`boto.ec2.autoscale.activity.Activity` instances.
        """
    name = autoscale_group
    if isinstance(autoscale_group, AutoScalingGroup):
        name = autoscale_group.name
    params = {'AutoScalingGroupName': name}
    if max_records:
        params['MaxRecords'] = max_records
    if next_token:
        params['NextToken'] = next_token
    if activity_ids:
        self.build_list_params(params, activity_ids, 'ActivityIds')
    return self.get_list('DescribeScalingActivities', params, [('member', Activity)])