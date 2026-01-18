import boto
import boto.jsonresponse
from boto.compat import json, six
from boto.resultset import ResultSet
from boto.iam.summarymap import SummaryMap
from boto.connection import AWSQueryConnection
def create_policy(self, policy_name, policy_document, path='/', description=None):
    """
        Create a policy.

        :type policy_name: string
        :param policy_name: The name of the new policy

        :type policy_document string
        :param policy_document: The document of the new policy

        :type path: string
        :param path: The path in which the policy will be created.
            Defaults to /.

        :type description: string
        :param path: A description of the new policy.

        """
    params = {'PolicyName': policy_name, 'PolicyDocument': policy_document, 'Path': path}
    if description is not None:
        params['Description'] = str(description)
    return self.get_response('CreatePolicy', params)