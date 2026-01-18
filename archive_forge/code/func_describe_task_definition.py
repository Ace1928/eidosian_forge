import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.ec2containerservice import exceptions
def describe_task_definition(self, task_definition):
    """
        Describes a task definition.

        :type task_definition: string
        :param task_definition: The `family` and `revision` (
            `family:revision`) or full Amazon Resource Name (ARN) of the task
            definition that you want to describe.

        """
    params = {'taskDefinition': task_definition}
    return self._make_request(action='DescribeTaskDefinition', verb='POST', path='/', params=params)