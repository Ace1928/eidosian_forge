import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.opsworks import exceptions
def describe_user_profiles(self, iam_user_arns=None):
    """
        Describe specified users.

        **Required Permissions**: To use this action, an IAM user must
        have an attached policy that explicitly grants permissions.
        For more information on user permissions, see `Managing User
        Permissions`_.

        :type iam_user_arns: list
        :param iam_user_arns: An array of IAM user ARNs that identify the users
            to be described.

        """
    params = {}
    if iam_user_arns is not None:
        params['IamUserArns'] = iam_user_arns
    return self.make_request(action='DescribeUserProfiles', body=json.dumps(params))