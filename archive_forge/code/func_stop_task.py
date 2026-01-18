import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.ec2containerservice import exceptions
def stop_task(self, task, cluster=None):
    """
        Stops a running task.

        :type cluster: string
        :param cluster: The short name or full Amazon Resource Name (ARN) of
            the cluster that hosts the task you want to stop. If you do not
            specify a cluster, the default cluster is assumed..

        :type task: string
        :param task: The task UUIDs or full Amazon Resource Name (ARN) entry of
            the task you would like to stop.

        """
    params = {'task': task}
    if cluster is not None:
        params['cluster'] = cluster
    return self._make_request(action='StopTask', verb='POST', path='/', params=params)