import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.opsworks import exceptions
def delete_user_profile(self, iam_user_arn):
    """
        Deletes a user profile.

        **Required Permissions**: To use this action, an IAM user must
        have an attached policy that explicitly grants permissions.
        For more information on user permissions, see `Managing User
        Permissions`_.

        :type iam_user_arn: string
        :param iam_user_arn: The user's IAM ARN.

        """
    params = {'IamUserArn': iam_user_arn}
    return self.make_request(action='DeleteUserProfile', body=json.dumps(params))