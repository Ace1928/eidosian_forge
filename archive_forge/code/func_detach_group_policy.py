import boto
import boto.jsonresponse
from boto.compat import json, six
from boto.resultset import ResultSet
from boto.iam.summarymap import SummaryMap
from boto.connection import AWSQueryConnection
def detach_group_policy(self, policy_arn, group_name):
    """
        :type policy_arn: string
        :param policy_arn: The ARN of the policy to detach

        :type group_name: string
        :param group_name: Group to detach the policy from

        """
    params = {'PolicyArn': policy_arn, 'GroupName': group_name}
    return self.get_response('DetachGroupPolicy', params)