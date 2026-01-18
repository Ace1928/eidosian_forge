import base64
import copy
import datetime
import json
import time
import oauth2client
from oauth2client import _helpers
from oauth2client import client
from oauth2client import crypt
from oauth2client import transport
@classmethod
def from_json_keyfile_name(cls, filename, scopes='', token_uri=None, revoke_uri=None):
    """Factory constructor from JSON keyfile by name.

        Args:
            filename: string, The location of the keyfile.
            scopes: List or string, (Optional) Scopes to use when acquiring an
                    access token.
            token_uri: string, URI for OAuth 2.0 provider token endpoint.
                       If unset and not present in the key file, defaults
                       to Google's endpoints.
            revoke_uri: string, URI for OAuth 2.0 provider revoke endpoint.
                       If unset and not present in the key file, defaults
                       to Google's endpoints.

        Returns:
            ServiceAccountCredentials, a credentials object created from
            the keyfile.

        Raises:
            ValueError, if the credential type is not :data:`SERVICE_ACCOUNT`.
            KeyError, if one of the expected keys is not present in
                the keyfile.
        """
    with open(filename, 'r') as file_obj:
        client_credentials = json.load(file_obj)
    return cls._from_parsed_json_keyfile(client_credentials, scopes, token_uri=token_uri, revoke_uri=revoke_uri)