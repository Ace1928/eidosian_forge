import boto
import boto.jsonresponse
from boto.compat import json, six
from boto.resultset import ResultSet
from boto.iam.summarymap import SummaryMap
from boto.connection import AWSQueryConnection
def get_credential_report(self):
    """
        Retrieves a credential report for an account

        A report must have been generated in the last 4 hours to succeed.
        The report is returned as a base64 encoded blob within the response.
        """
    params = {}
    return self.get_response('GetCredentialReport', params)