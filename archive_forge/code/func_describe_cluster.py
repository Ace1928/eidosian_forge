import types
import boto
import boto.utils
from boto.ec2.regioninfo import RegionInfo
from boto.emr.emrobject import AddInstanceGroupsResponse, BootstrapActionList, \
from boto.emr.step import JarStep
from boto.connection import AWSQueryConnection
from boto.exception import EmrResponseError
from boto.compat import six
def describe_cluster(self, cluster_id):
    """
        Describes an Elastic MapReduce cluster

        :type cluster_id: str
        :param cluster_id: The cluster id of interest
        """
    params = {'ClusterId': cluster_id}
    return self.get_object('DescribeCluster', params, Cluster)