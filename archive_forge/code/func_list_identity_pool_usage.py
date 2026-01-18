from boto.compat import json
from boto.exception import JSONResponseError
from boto.connection import AWSAuthConnection
from boto.regioninfo import RegionInfo
from boto.cognito.sync import exceptions
def list_identity_pool_usage(self, next_token=None, max_results=None):
    """
        Gets a list of identity pools registered with Cognito.

        :type next_token: string
        :param next_token: A pagination token for obtaining the next page of
            results.

        :type max_results: integer
        :param max_results: The maximum number of results to be returned.

        """
    uri = '/identitypools'
    params = {}
    headers = {}
    query_params = {}
    if next_token is not None:
        query_params['nextToken'] = next_token
    if max_results is not None:
        query_params['maxResults'] = max_results
    return self.make_request('GET', uri, expected_status=200, data=json.dumps(params), headers=headers, params=query_params)