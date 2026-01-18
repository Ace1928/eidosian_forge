import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.kms import exceptions
from boto.compat import six
import base64
def retire_grant(self, grant_token):
    """
        Retires a grant. You can retire a grant when you're done using
        it to clean up. You should revoke a grant when you intend to
        actively deny operations that depend on it.

        :type grant_token: string
        :param grant_token: Token that identifies the grant to be retired.

        """
    params = {'GrantToken': grant_token}
    return self.make_request(action='RetireGrant', body=json.dumps(params))