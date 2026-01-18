import types
import boto
import boto.utils
from boto.ec2.regioninfo import RegionInfo
from boto.emr.emrobject import AddInstanceGroupsResponse, BootstrapActionList, \
from boto.emr.step import JarStep
from boto.connection import AWSQueryConnection
from boto.exception import EmrResponseError
from boto.compat import six
def set_visible_to_all_users(self, jobflow_id, visibility):
    """
        Set whether specified Elastic Map Reduce job flows are visible to all IAM users

        :type jobflow_ids: list or str
        :param jobflow_ids: A list of job flow IDs

        :type visibility: bool
        :param visibility: Visibility
        """
    assert visibility in (True, False)
    params = {}
    params['VisibleToAllUsers'] = visibility and 'true' or 'false'
    self.build_list_params(params, [jobflow_id], 'JobFlowIds.member')
    return self.get_status('SetVisibleToAllUsers', params, verb='POST')