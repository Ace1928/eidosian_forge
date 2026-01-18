import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.kms import exceptions
from boto.compat import six
import base64
def revoke_grant(self, key_id, grant_id):
    """
        Revokes a grant. You can revoke a grant to actively deny
        operations that depend on it.

        :type key_id: string
        :param key_id: Unique identifier of the key associated with the grant.

        :type grant_id: string
        :param grant_id: Identifier of the grant to be revoked.

        """
    params = {'KeyId': key_id, 'GrantId': grant_id}
    return self.make_request(action='RevokeGrant', body=json.dumps(params))