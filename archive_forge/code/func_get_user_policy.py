import boto
import boto.jsonresponse
from boto.compat import json, six
from boto.resultset import ResultSet
from boto.iam.summarymap import SummaryMap
from boto.connection import AWSQueryConnection
def get_user_policy(self, user_name, policy_name):
    """
        Retrieves the specified policy document for the specified user.

        :type user_name: string
        :param user_name: The name of the user the policy is associated with.

        :type policy_name: string
        :param policy_name: The policy document to get.

        """
    params = {'UserName': user_name, 'PolicyName': policy_name}
    return self.get_response('GetUserPolicy', params, verb='POST')