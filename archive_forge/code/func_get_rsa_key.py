from __future__ import absolute_import, unicode_literals
import sys
from . import SIGNATURE_METHODS, utils
def get_rsa_key(self, client_key, request):
    """Retrieves a previously stored client provided RSA key.

        :param client_key: The client/consumer key.
        :param request: OAuthlib request.
        :type request: oauthlib.common.Request
        :returns: The rsa public key as a string.

        This method must allow the use of a dummy client_key value. Fetching
        the rsa key using the dummy key must take the same amount of time
        as fetching a key for a valid client. The dummy key must also be of
        the same bit length as client keys.

        Note that the key must be returned in plaintext.

        This method is used by

        * AccessTokenEndpoint
        * RequestTokenEndpoint
        * ResourceEndpoint
        * SignatureOnlyEndpoint
        """
    raise self._subclass_must_implement('get_rsa_key')