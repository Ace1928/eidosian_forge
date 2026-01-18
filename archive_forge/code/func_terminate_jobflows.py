import types
import boto
import boto.utils
from boto.ec2.regioninfo import RegionInfo
from boto.emr.emrobject import AddInstanceGroupsResponse, BootstrapActionList, \
from boto.emr.step import JarStep
from boto.connection import AWSQueryConnection
from boto.exception import EmrResponseError
from boto.compat import six
def terminate_jobflows(self, jobflow_ids):
    """
        Terminate an Elastic MapReduce job flow

        :type jobflow_ids: list
        :param jobflow_ids: A list of job flow IDs
        """
    params = {}
    self.build_list_params(params, jobflow_ids, 'JobFlowIds.member')
    return self.get_status('TerminateJobFlows', params, verb='POST')