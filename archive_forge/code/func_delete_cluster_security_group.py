import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.redshift import exceptions
def delete_cluster_security_group(self, cluster_security_group_name):
    """
        Deletes an Amazon Redshift security group.

        For information about managing security groups, go to `Amazon
        Redshift Cluster Security Groups`_ in the Amazon Redshift
        Management Guide .

        :type cluster_security_group_name: string
        :param cluster_security_group_name: The name of the cluster security
            group to be deleted.

        """
    params = {'ClusterSecurityGroupName': cluster_security_group_name}
    return self._make_request(action='DeleteClusterSecurityGroup', verb='POST', path='/', params=params)