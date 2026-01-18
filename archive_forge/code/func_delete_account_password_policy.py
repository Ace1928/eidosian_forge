import boto
import boto.jsonresponse
from boto.compat import json, six
from boto.resultset import ResultSet
from boto.iam.summarymap import SummaryMap
from boto.connection import AWSQueryConnection
def delete_account_password_policy(self):
    """
        Delete the password policy currently set for the AWS account.
        """
    params = {}
    return self.get_response('DeleteAccountPasswordPolicy', params)