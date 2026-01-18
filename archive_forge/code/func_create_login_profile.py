import boto
import boto.jsonresponse
from boto.compat import json, six
from boto.resultset import ResultSet
from boto.iam.summarymap import SummaryMap
from boto.connection import AWSQueryConnection
def create_login_profile(self, user_name, password):
    """
        Creates a login profile for the specified user, give the user the
        ability to access AWS services and the AWS Management Console.

        :type user_name: string
        :param user_name: The name of the user

        :type password: string
        :param password: The new password for the user

        """
    params = {'UserName': user_name, 'Password': password}
    return self.get_response('CreateLoginProfile', params)