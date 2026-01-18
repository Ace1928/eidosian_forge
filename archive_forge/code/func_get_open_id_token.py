import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.cognito.identity import exceptions
def get_open_id_token(self, identity_id, logins=None):
    """
        Gets an OpenID token, using a known Cognito ID. This known
        Cognito ID is returned by GetId. You can optionally add
        additional logins for the identity. Supplying multiple logins
        creates an implicit link.

        The OpenId token is valid for 15 minutes.

        :type identity_id: string
        :param identity_id: A unique identifier in the format REGION:GUID.

        :type logins: map
        :param logins: A set of optional name-value pairs that map provider
            names to provider tokens.

        """
    params = {'IdentityId': identity_id}
    if logins is not None:
        params['Logins'] = logins
    return self.make_request(action='GetOpenIdToken', body=json.dumps(params))