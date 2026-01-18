from __future__ import absolute_import, unicode_literals
import sys
from . import SIGNATURE_METHODS, utils
@property
def dummy_access_token(self):
    """Dummy access token used when an invalid token was supplied.

        :returns: The dummy access token string.

        The dummy access token should be associated with an access token
        secret such that get_access_token_secret(.., dummy_access_token)
        returns a valid secret.

        This method is used by

        * ResourceEndpoint
        """
    raise self._subclass_must_implement('dummy_access_token')