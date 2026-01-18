import types
import boto
import boto.utils
from boto.ec2.regioninfo import RegionInfo
from boto.emr.emrobject import AddInstanceGroupsResponse, BootstrapActionList, \
from boto.emr.step import JarStep
from boto.connection import AWSQueryConnection
from boto.exception import EmrResponseError
from boto.compat import six
def describe_jobflows(self, states=None, jobflow_ids=None, created_after=None, created_before=None):
    """
        This method is deprecated. We recommend you use list_clusters,
        describe_cluster, list_steps, list_instance_groups and
        list_bootstrap_actions instead.

        Retrieve all the Elastic MapReduce job flows on your account

        :type states: list
        :param states: A list of strings with job flow states wanted

        :type jobflow_ids: list
        :param jobflow_ids: A list of job flow IDs
        :type created_after: datetime
        :param created_after: Bound on job flow creation time

        :type created_before: datetime
        :param created_before: Bound on job flow creation time
        """
    params = {}
    if states:
        self.build_list_params(params, states, 'JobFlowStates.member')
    if jobflow_ids:
        self.build_list_params(params, jobflow_ids, 'JobFlowIds.member')
    if created_after:
        params['CreatedAfter'] = created_after.strftime(boto.utils.ISO8601)
    if created_before:
        params['CreatedBefore'] = created_before.strftime(boto.utils.ISO8601)
    return self.get_list('DescribeJobFlows', params, [('member', JobFlow)])