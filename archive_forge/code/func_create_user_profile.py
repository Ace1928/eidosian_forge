import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.opsworks import exceptions
def create_user_profile(self, iam_user_arn, ssh_username=None, ssh_public_key=None, allow_self_management=None):
    """
        Creates a new user profile.

        **Required Permissions**: To use this action, an IAM user must
        have an attached policy that explicitly grants permissions.
        For more information on user permissions, see `Managing User
        Permissions`_.

        :type iam_user_arn: string
        :param iam_user_arn: The user's IAM ARN.

        :type ssh_username: string
        :param ssh_username: The user's SSH user name. The allowable characters
            are [a-z], [A-Z], [0-9], '-', and '_'. If the specified name
            includes other punctuation marks, AWS OpsWorks removes them. For
            example, `my.name` will be changed to `myname`. If you do not
            specify an SSH user name, AWS OpsWorks generates one from the IAM
            user name.

        :type ssh_public_key: string
        :param ssh_public_key: The user's public SSH key.

        :type allow_self_management: boolean
        :param allow_self_management: Whether users can specify their own SSH
            public key through the My Settings page. For more information, see
            `Setting an IAM User's Public SSH Key`_.

        """
    params = {'IamUserArn': iam_user_arn}
    if ssh_username is not None:
        params['SshUsername'] = ssh_username
    if ssh_public_key is not None:
        params['SshPublicKey'] = ssh_public_key
    if allow_self_management is not None:
        params['AllowSelfManagement'] = allow_self_management
    return self.make_request(action='CreateUserProfile', body=json.dumps(params))