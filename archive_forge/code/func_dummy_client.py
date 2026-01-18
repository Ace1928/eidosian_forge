from __future__ import absolute_import, unicode_literals
import sys
from . import SIGNATURE_METHODS, utils
@property
def dummy_client(self):
    """Dummy client used when an invalid client key is supplied.

        :returns: The dummy client key string.

        The dummy client should be associated with either a client secret,
        a rsa key or both depending on which signature methods are supported.
        Providers should make sure that

        get_client_secret(dummy_client)
        get_rsa_key(dummy_client)

        return a valid secret or key for the dummy client.

        This method is used by

        * AccessTokenEndpoint
        * RequestTokenEndpoint
        * ResourceEndpoint
        * SignatureOnlyEndpoint
        """
    raise self._subclass_must_implement('dummy_client')