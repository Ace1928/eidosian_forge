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
def get_all_policies(self, as_group=None, policy_names=None, max_records=None, next_token=None):
    """
        Returns descriptions of what each policy does. This action supports
        pagination. If the response includes a token, there are more records
        available. To get the additional records, repeat the request with the
        response token as the NextToken parameter.

        If no group name or list of policy names are provided, all
        available policies are returned.

        :type as_group: str
        :param as_group: The name of the
            :class:`boto.ec2.autoscale.group.AutoScalingGroup` to filter for.

        :type policy_names: list
        :param policy_names: List of policy names which should be searched for.

        :type max_records: int
        :param max_records: Maximum amount of groups to return.

        :type next_token: str
        :param next_token: If you have more results than can be returned
            at once, pass in this  parameter to page through all results.
        """
    params = {}
    if as_group:
        params['AutoScalingGroupName'] = as_group
    if policy_names:
        self.build_list_params(params, policy_names, 'PolicyNames')
    if max_records:
        params['MaxRecords'] = max_records
    if next_token:
        params['NextToken'] = next_token
    return self.get_list('DescribePolicies', params, [('member', ScalingPolicy)])