import boto
import boto.jsonresponse
from boto.compat import json, six
from boto.resultset import ResultSet
from boto.iam.summarymap import SummaryMap
from boto.connection import AWSQueryConnection
def attach_user_policy(self, policy_arn, user_name):
    """
        :type policy_arn: string
        :param policy_arn: The ARN of the policy to attach

        :type user_name: string
        :param user_name: User to attach the policy to

        """
    params = {'PolicyArn': policy_arn, 'UserName': user_name}
    return self.get_response('AttachUserPolicy', params)