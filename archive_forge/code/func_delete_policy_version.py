import boto
import boto.jsonresponse
from boto.compat import json, six
from boto.resultset import ResultSet
from boto.iam.summarymap import SummaryMap
from boto.connection import AWSQueryConnection
def delete_policy_version(self, policy_arn, version_id):
    """
        Delete a policy version.

        :type policy_arn: string
        :param policy_arn: The ARN of the policy to delete a version from

        :type version_id: string
        :param version_id: The id of the version to delete

        """
    params = {'PolicyArn': policy_arn, 'VersionId': version_id}
    return self.get_response('DeletePolicyVersion', params)