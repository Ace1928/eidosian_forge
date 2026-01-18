import boto
import boto.jsonresponse
from boto.compat import json, six
from boto.resultset import ResultSet
from boto.iam.summarymap import SummaryMap
from boto.connection import AWSQueryConnection
def delete_user(self, user_name):
    """
        Delete a user including the user's path, GUID and ARN.

        If the user_name is not specified, the user_name is determined
        implicitly based on the AWS Access Key ID used to sign the request.

        :type user_name: string
        :param user_name: The name of the user to delete.

        """
    params = {'UserName': user_name}
    return self.get_response('DeleteUser', params)