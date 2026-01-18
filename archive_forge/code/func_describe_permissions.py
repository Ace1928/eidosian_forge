import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.opsworks import exceptions
def describe_permissions(self, iam_user_arn=None, stack_id=None):
    """
        Describes the permissions for a specified stack.

        **Required Permissions**: To use this action, an IAM user must
        have a Manage permissions level for the stack, or an attached
        policy that explicitly grants permissions. For more
        information on user permissions, see `Managing User
        Permissions`_.

        :type iam_user_arn: string
        :param iam_user_arn: The user's IAM ARN. For more information about IAM
            ARNs, see `Using Identifiers`_.

        :type stack_id: string
        :param stack_id: The stack ID.

        """
    params = {}
    if iam_user_arn is not None:
        params['IamUserArn'] = iam_user_arn
    if stack_id is not None:
        params['StackId'] = stack_id
    return self.make_request(action='DescribePermissions', body=json.dumps(params))