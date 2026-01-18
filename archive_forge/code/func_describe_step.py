import types
import boto
import boto.utils
from boto.ec2.regioninfo import RegionInfo
from boto.emr.emrobject import AddInstanceGroupsResponse, BootstrapActionList, \
from boto.emr.step import JarStep
from boto.connection import AWSQueryConnection
from boto.exception import EmrResponseError
from boto.compat import six
def describe_step(self, cluster_id, step_id):
    """
        Describe an Elastic MapReduce step

        :type cluster_id: str
        :param cluster_id: The cluster id of interest
        :type step_id: str
        :param step_id: The step id of interest
        """
    params = {'ClusterId': cluster_id, 'StepId': step_id}
    return self.get_object('DescribeStep', params, HadoopStep)