import boto
import boto.jsonresponse
from boto.compat import json, six
from boto.resultset import ResultSet
from boto.iam.summarymap import SummaryMap
from boto.connection import AWSQueryConnection
def detach_user_policy(self, policy_arn, user_name):
    """
        :type policy_arn: string
        :param policy_arn: The ARN of the policy to detach

        :type user_name: string
        :param user_name: User to detach the policy from

        """
    params = {'PolicyArn': policy_arn, 'UserName': user_name}
    return self.get_response('DetachUserPolicy', params)