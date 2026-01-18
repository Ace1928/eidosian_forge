import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.redshift import exceptions
def delete_cluster_parameter_group(self, parameter_group_name):
    """
        Deletes a specified Amazon Redshift parameter group.

        :type parameter_group_name: string
        :param parameter_group_name:
        The name of the parameter group to be deleted.

        Constraints:


        + Must be the name of an existing cluster parameter group.
        + Cannot delete a default cluster parameter group.

        """
    params = {'ParameterGroupName': parameter_group_name}
    return self._make_request(action='DeleteClusterParameterGroup', verb='POST', path='/', params=params)