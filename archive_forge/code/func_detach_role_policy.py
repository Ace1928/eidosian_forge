import boto
import boto.jsonresponse
from boto.compat import json, six
from boto.resultset import ResultSet
from boto.iam.summarymap import SummaryMap
from boto.connection import AWSQueryConnection
def detach_role_policy(self, policy_arn, role_name):
    """
        :type policy_arn: string
        :param policy_arn: The ARN of the policy to detach

        :type role_name: string
        :param role_name: Role to detach the policy from

        """
    params = {'PolicyArn': policy_arn, 'RoleName': role_name}
    return self.get_response('DetachRolePolicy', params)