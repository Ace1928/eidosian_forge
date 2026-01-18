import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.redshift import exceptions
def create_cluster_parameter_group(self, parameter_group_name, parameter_group_family, description):
    """
        Creates an Amazon Redshift parameter group.

        Creating parameter groups is independent of creating clusters.
        You can associate a cluster with a parameter group when you
        create the cluster. You can also associate an existing cluster
        with a parameter group after the cluster is created by using
        ModifyCluster.

        Parameters in the parameter group define specific behavior
        that applies to the databases you create on the cluster. For
        more information about managing parameter groups, go to
        `Amazon Redshift Parameter Groups`_ in the Amazon Redshift
        Management Guide .

        :type parameter_group_name: string
        :param parameter_group_name:
        The name of the cluster parameter group.

        Constraints:


        + Must be 1 to 255 alphanumeric characters or hyphens
        + First character must be a letter.
        + Cannot end with a hyphen or contain two consecutive hyphens.
        + Must be unique within your AWS account.

        This value is stored as a lower-case string.

        :type parameter_group_family: string
        :param parameter_group_family: The Amazon Redshift engine version to
            which the cluster parameter group applies. The cluster engine
            version determines the set of parameters.
        To get a list of valid parameter group family names, you can call
            DescribeClusterParameterGroups. By default, Amazon Redshift returns
            a list of all the parameter groups that are owned by your AWS
            account, including the default parameter groups for each Amazon
            Redshift engine version. The parameter group family names
            associated with the default parameter groups provide you the valid
            values. For example, a valid family name is "redshift-1.0".

        :type description: string
        :param description: A description of the parameter group.

        """
    params = {'ParameterGroupName': parameter_group_name, 'ParameterGroupFamily': parameter_group_family, 'Description': description}
    return self._make_request(action='CreateClusterParameterGroup', verb='POST', path='/', params=params)