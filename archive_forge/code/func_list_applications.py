import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.codedeploy import exceptions
def list_applications(self, next_token=None):
    """
        Lists the applications registered within the AWS user account.

        :type next_token: string
        :param next_token: An identifier that was returned from the previous
            list applications call, which can be used to return the next set of
            applications in the list.

        """
    params = {}
    if next_token is not None:
        params['nextToken'] = next_token
    return self.make_request(action='ListApplications', body=json.dumps(params))