import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.cognito.identity import exceptions
def list_identity_pools(self, max_results, next_token=None):
    """
        Lists all of the Cognito identity pools registered for your
        account.

        :type max_results: integer
        :param max_results: The maximum number of identities to return.

        :type next_token: string
        :param next_token: A pagination token.

        """
    params = {'MaxResults': max_results}
    if next_token is not None:
        params['NextToken'] = next_token
    return self.make_request(action='ListIdentityPools', body=json.dumps(params))