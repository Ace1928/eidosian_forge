import boto
import boto.jsonresponse
from boto.compat import json, six
from boto.resultset import ResultSet
from boto.iam.summarymap import SummaryMap
from boto.connection import AWSQueryConnection
def set_default_policy_version(self, policy_arn, version_id):
    """
        Set default policy version.

        :type policy_arn: string
        :param policy_arn: The ARN of the policy to set the default version
            for

        :type version_id: string
        :param version_id: The id of the version to set as default
        """
    params = {'PolicyArn': policy_arn, 'VersionId': version_id}
    return self.get_response('SetDefaultPolicyVersion', params)