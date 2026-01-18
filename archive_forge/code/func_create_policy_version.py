import boto
import boto.jsonresponse
from boto.compat import json, six
from boto.resultset import ResultSet
from boto.iam.summarymap import SummaryMap
from boto.connection import AWSQueryConnection
def create_policy_version(self, policy_arn, policy_document, set_as_default=None):
    """
        Create a policy version.

        :type policy_arn: string
        :param policy_arn: The ARN of the policy

        :type policy_document string
        :param policy_document: The document of the new policy version

        :type set_as_default: bool
        :param set_as_default: Sets the policy version as default
            Defaults to None.

        """
    params = {'PolicyArn': policy_arn, 'PolicyDocument': policy_document}
    if type(set_as_default) == bool:
        params['SetAsDefault'] = str(set_as_default).lower()
    return self.get_response('CreatePolicyVersion', params)