import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.ec2containerservice import exceptions
def register_task_definition(self, family, container_definitions):
    """
        Registers a new task definition from the supplied `family` and
        `containerDefinitions`.

        :type family: string
        :param family: You can specify a `family` for a task definition, which
            allows you to track multiple versions of the same task definition.
            You can think of the `family` as a name for your task definition.

        :type container_definitions: list
        :param container_definitions: A list of container definitions in JSON
            format that describe the different containers that make up your
            task.

        """
    params = {'family': family}
    self.build_complex_list_params(params, container_definitions, 'containerDefinitions.member', ('name', 'image', 'cpu', 'memory', 'links', 'portMappings', 'essential', 'entryPoint', 'command', 'environment'))
    return self._make_request(action='RegisterTaskDefinition', verb='POST', path='/', params=params)