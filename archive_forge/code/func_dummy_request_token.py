from __future__ import absolute_import, unicode_literals
import sys
from . import SIGNATURE_METHODS, utils
@property
def dummy_request_token(self):
    """Dummy request token used when an invalid token was supplied.

        :returns: The dummy request token string.

        The dummy request token should be associated with a request token
        secret such that get_request_token_secret(.., dummy_request_token)
        returns a valid secret.

        This method is used by

        * AccessTokenEndpoint
        """
    raise self._subclass_must_implement('dummy_request_token')