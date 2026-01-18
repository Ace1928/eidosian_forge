import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.ec2containerservice import exceptions
def run_task(self, task_definition, cluster=None, overrides=None, count=None):
    """
        Start a task using random placement and the default Amazon ECS
        scheduler. If you want to use your own scheduler or place a
        task on a specific container instance, use `StartTask`
        instead.

        :type cluster: string
        :param cluster: The short name or full Amazon Resource Name (ARN) of
            the cluster that you want to run your task on. If you do not
            specify a cluster, the default cluster is assumed..

        :type task_definition: string
        :param task_definition: The `family` and `revision` (
            `family:revision`) or full Amazon Resource Name (ARN) of the task
            definition that you want to run.

        :type overrides: dict
        :param overrides:

        :type count: integer
        :param count: The number of instances of the specified task that you
            would like to place on your cluster.

        """
    params = {'taskDefinition': task_definition}
    if cluster is not None:
        params['cluster'] = cluster
    if overrides is not None:
        params['overrides'] = overrides
    if count is not None:
        params['count'] = count
    return self._make_request(action='RunTask', verb='POST', path='/', params=params)