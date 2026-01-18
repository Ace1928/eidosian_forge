from __future__ import absolute_import, unicode_literals
import logging
def authenticate_client_id(self, client_id, request, *args, **kwargs):
    """Ensure client_id belong to a non-confidential client.

        A non-confidential client is one that is not required to authenticate
        through other means, such as using HTTP Basic.

        Note, while not strictly necessary it can often be very convenient
        to set request.client to the client object associated with the
        given client_id.

        :param client_id: Unicode client identifier.
        :param request: OAuthlib request.
        :type request: oauthlib.common.Request
        :rtype: True or False

        Method is used by:
            - Authorization Code Grant
        """
    raise NotImplementedError('Subclasses must implement this method.')