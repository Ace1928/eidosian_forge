import types
import boto
import boto.utils
from boto.ec2.regioninfo import RegionInfo
from boto.emr.emrobject import AddInstanceGroupsResponse, BootstrapActionList, \
from boto.emr.step import JarStep
from boto.connection import AWSQueryConnection
from boto.exception import EmrResponseError
from boto.compat import six
def list_steps(self, cluster_id, step_states=None, marker=None):
    """
        List cluster steps

        :type cluster_id: str
        :param cluster_id: The cluster id of interest
        :type step_states: list
        :param step_states: Filter by step states
        :type marker: str
        :param marker: Pagination marker
        """
    params = {'ClusterId': cluster_id}
    if marker:
        params['Marker'] = marker
    if step_states:
        self.build_list_params(params, step_states, 'StepStates.member')
    return self.get_object('ListSteps', params, StepSummaryList)