import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.ec2containerservice import exceptions
def describe_tasks(self, tasks, cluster=None):
    """
        Describes a specified task or tasks.

        :type cluster: string
        :param cluster: The short name or full Amazon Resource Name (ARN) of
            the cluster that hosts the task you want to describe. If you do not
            specify a cluster, the default cluster is assumed.

        :type tasks: list
        :param tasks: A space-separated list of task UUIDs or full Amazon
            Resource Name (ARN) entries.

        """
    params = {}
    self.build_list_params(params, tasks, 'tasks.member')
    if cluster is not None:
        params['cluster'] = cluster
    return self._make_request(action='DescribeTasks', verb='POST', path='/', params=params)