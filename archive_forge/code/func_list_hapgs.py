import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.cloudhsm import exceptions
def list_hapgs(self, next_token=None):
    """
        Lists the high-availability partition groups for the account.

        This operation supports pagination with the use of the
        NextToken member. If more results are available, the NextToken
        member of the response contains a token that you pass in the
        next call to ListHapgs to retrieve the next set of items.

        :type next_token: string
        :param next_token: The NextToken value from a previous call to
            ListHapgs. Pass null if this is the first call.

        """
    params = {}
    if next_token is not None:
        params['NextToken'] = next_token
    return self.make_request(action='ListHapgs', body=json.dumps(params))