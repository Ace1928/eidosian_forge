import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.opsworks import exceptions
def describe_service_errors(self, stack_id=None, instance_id=None, service_error_ids=None):
    """
        Describes AWS OpsWorks service errors.

        **Required Permissions**: To use this action, an IAM user must
        have a Show, Deploy, or Manage permissions level for the
        stack, or an attached policy that explicitly grants
        permissions. For more information on user permissions, see
        `Managing User Permissions`_.

        :type stack_id: string
        :param stack_id: The stack ID. If you use this parameter,
            `DescribeServiceErrors` returns descriptions of the errors
            associated with the specified stack.

        :type instance_id: string
        :param instance_id: The instance ID. If you use this parameter,
            `DescribeServiceErrors` returns descriptions of the errors
            associated with the specified instance.

        :type service_error_ids: list
        :param service_error_ids: An array of service error IDs. If you use
            this parameter, `DescribeServiceErrors` returns descriptions of the
            specified errors. Otherwise, it returns a description of every
            error.

        """
    params = {}
    if stack_id is not None:
        params['StackId'] = stack_id
    if instance_id is not None:
        params['InstanceId'] = instance_id
    if service_error_ids is not None:
        params['ServiceErrorIds'] = service_error_ids
    return self.make_request(action='DescribeServiceErrors', body=json.dumps(params))