import types
import boto
import boto.utils
from boto.ec2.regioninfo import RegionInfo
from boto.emr.emrobject import AddInstanceGroupsResponse, BootstrapActionList, \
from boto.emr.step import JarStep
from boto.connection import AWSQueryConnection
from boto.exception import EmrResponseError
from boto.compat import six
def set_termination_protection(self, jobflow_id, termination_protection_status):
    """
        Set termination protection on specified Elastic MapReduce job flows

        :type jobflow_ids: list or str
        :param jobflow_ids: A list of job flow IDs

        :type termination_protection_status: bool
        :param termination_protection_status: Termination protection status
        """
    assert termination_protection_status in (True, False)
    params = {}
    params['TerminationProtected'] = termination_protection_status and 'true' or 'false'
    self.build_list_params(params, [jobflow_id], 'JobFlowIds.member')
    return self.get_status('SetTerminationProtection', params, verb='POST')