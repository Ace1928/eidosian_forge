from __future__ import absolute_import, unicode_literals
import sys
from . import SIGNATURE_METHODS, utils
def get_default_realms(self, client_key, request):
    """Get the default realms for a client.

        :param client_key: The client/consumer key.
        :param request: OAuthlib request.
        :type request: oauthlib.common.Request
        :returns: The list of default realms associated with the client.

        The list of default realms will be set during client registration and
        is outside the scope of OAuthLib.

        This method is used by

        * RequestTokenEndpoint
        """
    raise self._subclass_must_implement('get_default_realms')