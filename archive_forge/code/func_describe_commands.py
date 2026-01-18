import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.opsworks import exceptions
def describe_commands(self, deployment_id=None, instance_id=None, command_ids=None):
    """
        Describes the results of specified commands.


        You must specify at least one of the parameters.


        **Required Permissions**: To use this action, an IAM user must
        have a Show, Deploy, or Manage permissions level for the
        stack, or an attached policy that explicitly grants
        permissions. For more information on user permissions, see
        `Managing User Permissions`_.

        :type deployment_id: string
        :param deployment_id: The deployment ID. If you include this parameter,
            `DescribeCommands` returns a description of the commands associated
            with the specified deployment.

        :type instance_id: string
        :param instance_id: The instance ID. If you include this parameter,
            `DescribeCommands` returns a description of the commands associated
            with the specified instance.

        :type command_ids: list
        :param command_ids: An array of command IDs. If you include this
            parameter, `DescribeCommands` returns a description of the
            specified commands. Otherwise, it returns a description of every
            command.

        """
    params = {}
    if deployment_id is not None:
        params['DeploymentId'] = deployment_id
    if instance_id is not None:
        params['InstanceId'] = instance_id
    if command_ids is not None:
        params['CommandIds'] = command_ids
    return self.make_request(action='DescribeCommands', body=json.dumps(params))