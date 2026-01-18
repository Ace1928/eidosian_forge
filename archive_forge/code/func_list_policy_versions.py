import boto
import boto.jsonresponse
from boto.compat import json, six
from boto.resultset import ResultSet
from boto.iam.summarymap import SummaryMap
from boto.connection import AWSQueryConnection
def list_policy_versions(self, policy_arn, marker=None, max_items=None):
    """
        List policy versions.

        :type policy_arn: string
        :param policy_arn: The ARN of the policy to get versions of

        :type marker: string
        :param marker: A marker used for pagination (received from previous
            accesses)

        :type max_items: int
        :param max_items: Send only max_items; allows paginations

        """
    params = {'PolicyArn': policy_arn}
    if marker is not None:
        params['Marker'] = marker
    if max_items is not None:
        params['MaxItems'] = max_items
    return self.get_response('ListPolicyVersions', params, list_marker='Versions')