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
def get_all_launch_configurations(self, **kwargs):
    """
        Returns a full description of the launch configurations given the
        specified names.

        If no names are specified, then the full details of all launch
        configurations are returned.

        :type names: list
        :param names: List of configuration names which should be searched for.

        :type max_records: int
        :param max_records: Maximum amount of configurations to return.

        :type next_token: str
        :param next_token: If you have more results than can be returned
            at once, pass in this  parameter to page through all results.

        :rtype: list
        :returns: List of
            :class:`boto.ec2.autoscale.launchconfig.LaunchConfiguration`
            instances.
        """
    params = {}
    max_records = kwargs.get('max_records', None)
    names = kwargs.get('names', None)
    if max_records is not None:
        params['MaxRecords'] = max_records
    if names:
        self.build_list_params(params, names, 'LaunchConfigurationNames')
    next_token = kwargs.get('next_token')
    if next_token:
        params['NextToken'] = next_token
    return self.get_list('DescribeLaunchConfigurations', params, [('member', LaunchConfiguration)])