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
def from_p12_keyfile(cls, service_account_email, filename, private_key_password=None, scopes='', token_uri=oauth2client.GOOGLE_TOKEN_URI, revoke_uri=oauth2client.GOOGLE_REVOKE_URI):
    """Factory constructor from JSON keyfile.

        Args:
            service_account_email: string, The email associated with the
                                   service account.
            filename: string, The location of the PKCS#12 keyfile.
            private_key_password: string, (Optional) Password for PKCS#12
                                  private key. Defaults to ``notasecret``.
            scopes: List or string, (Optional) Scopes to use when acquiring an
                    access token.
            token_uri: string, URI for token endpoint. For convenience defaults
                       to Google's endpoints but any OAuth 2.0 provider can be
                       used.
            revoke_uri: string, URI for revoke endpoint. For convenience
                        defaults to Google's endpoints but any OAuth 2.0
                        provider can be used.

        Returns:
            ServiceAccountCredentials, a credentials object created from
            the keyfile.

        Raises:
            NotImplementedError if pyOpenSSL is not installed / not the
            active crypto library.
        """
    with open(filename, 'rb') as file_obj:
        private_key_pkcs12 = file_obj.read()
    return cls._from_p12_keyfile_contents(service_account_email, private_key_pkcs12, private_key_password=private_key_password, scopes=scopes, token_uri=token_uri, revoke_uri=revoke_uri)