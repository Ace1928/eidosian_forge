import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.kms import exceptions
from boto.compat import six
import base64
def create_grant(self, key_id, grantee_principal, retiring_principal=None, operations=None, constraints=None, grant_tokens=None):
    """
        Adds a grant to a key to specify who can access the key and
        under what conditions. Grants are alternate permission
        mechanisms to key policies. If absent, access to the key is
        evaluated based on IAM policies attached to the user. By
        default, grants do not expire. Grants can be listed, retired,
        or revoked as indicated by the following APIs. Typically, when
        you are finished using a grant, you retire it. When you want
        to end a grant immediately, revoke it. For more information
        about grants, see `Grants`_.

        #. ListGrants
        #. RetireGrant
        #. RevokeGrant

        :type key_id: string
        :param key_id: A unique key identifier for a customer master key. This
            value can be a globally unique identifier, an ARN, or an alias.

        :type grantee_principal: string
        :param grantee_principal: Principal given permission by the grant to
            use the key identified by the `keyId` parameter.

        :type retiring_principal: string
        :param retiring_principal: Principal given permission to retire the
            grant. For more information, see RetireGrant.

        :type operations: list
        :param operations: List of operations permitted by the grant. This can
            be any combination of one or more of the following values:

        #. Decrypt
        #. Encrypt
        #. GenerateDataKey
        #. GenerateDataKeyWithoutPlaintext
        #. ReEncryptFrom
        #. ReEncryptTo
        #. CreateGrant

        :type constraints: dict
        :param constraints: Specifies the conditions under which the actions
            specified by the `Operations` parameter are allowed.

        :type grant_tokens: list
        :param grant_tokens: List of grant tokens.

        """
    params = {'KeyId': key_id, 'GranteePrincipal': grantee_principal}
    if retiring_principal is not None:
        params['RetiringPrincipal'] = retiring_principal
    if operations is not None:
        params['Operations'] = operations
    if constraints is not None:
        params['Constraints'] = constraints
    if grant_tokens is not None:
        params['GrantTokens'] = grant_tokens
    return self.make_request(action='CreateGrant', body=json.dumps(params))