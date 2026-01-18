from __future__ import absolute_import, unicode_literals
import sys
from . import SIGNATURE_METHODS, utils
def get_request_token_secret(self, client_key, token, request):
    """Retrieves the shared secret associated with the request token.

        :param client_key: The client/consumer key.
        :param token: The request token string.
        :param request: OAuthlib request.
        :type request: oauthlib.common.Request
        :returns: The token secret as a string.

        This method must allow the use of a dummy values and the running time
        must be roughly equivalent to that of the running time of valid values::

            # Unlikely to be near constant time as it uses two database
            # lookups for a valid client, and only one for an invalid.
            from your_datastore import RequestTokenSecret
            if RequestTokenSecret.has(client_key):
                return RequestTokenSecret.get((client_key, request_token))
            else:
                return 'dummy'

            # Aim to mimic number of latency inducing operations no matter
            # whether the client is valid or not.
            from your_datastore import RequestTokenSecret
            return ClientSecret.get((client_key, request_token), 'dummy')

        Note that the returned key must be in plaintext.

        This method is used by

        * AccessTokenEndpoint
        """
    raise self._subclass_must_implement('get_request_token_secret')