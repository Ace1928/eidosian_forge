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
def create_scheduled_group_action(self, as_group, name, time=None, desired_capacity=None, min_size=None, max_size=None, start_time=None, end_time=None, recurrence=None):
    """
        Creates a scheduled scaling action for a Auto Scaling group. If you
        leave a parameter unspecified, the corresponding value remains
        unchanged in the affected Auto Scaling group.

        :type as_group: string
        :param as_group: The auto scaling group to get activities on.

        :type name: string
        :param name: Scheduled action name.

        :type time: datetime.datetime
        :param time: The time for this action to start. (Depracated)

        :type desired_capacity: int
        :param desired_capacity: The number of EC2 instances that should
            be running in this group.

        :type min_size: int
        :param min_size: The minimum size for the new auto scaling group.

        :type max_size: int
        :param max_size: The minimum size for the new auto scaling group.

        :type start_time: datetime.datetime
        :param start_time: The time for this action to start. When StartTime and EndTime are specified with Recurrence, they form the boundaries of when the recurring action will start and stop.

        :type end_time: datetime.datetime
        :param end_time: The time for this action to end. When StartTime and EndTime are specified with Recurrence, they form the boundaries of when the recurring action will start and stop.

        :type recurrence: string
        :param recurrence: The time when recurring future actions will start. Start time is specified by the user following the Unix cron syntax format. EXAMPLE: '0 10 * * *'
        """
    params = {'AutoScalingGroupName': as_group, 'ScheduledActionName': name}
    if start_time is not None:
        params['StartTime'] = start_time.isoformat()
    if end_time is not None:
        params['EndTime'] = end_time.isoformat()
    if recurrence is not None:
        params['Recurrence'] = recurrence
    if time:
        params['Time'] = time.isoformat()
    if desired_capacity is not None:
        params['DesiredCapacity'] = desired_capacity
    if min_size is not None:
        params['MinSize'] = min_size
    if max_size is not None:
        params['MaxSize'] = max_size
    return self.get_status('PutScheduledUpdateGroupAction', params)