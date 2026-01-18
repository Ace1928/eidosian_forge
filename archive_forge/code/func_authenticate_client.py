from __future__ import absolute_import, unicode_literals
import logging
def authenticate_client(self, request, *args, **kwargs):
    """Authenticate client through means outside the OAuth 2 spec.

        Means of authentication is negotiated beforehand and may for example
        be `HTTP Basic Authentication Scheme`_ which utilizes the Authorization
        header.

        Headers may be accesses through request.headers and parameters found in
        both body and query can be obtained by direct attribute access, i.e.
        request.client_id for client_id in the URL query.

        :param request: OAuthlib request.
        :type request: oauthlib.common.Request
        :rtype: True or False

        Method is used by:
            - Authorization Code Grant
            - Resource Owner Password Credentials Grant (may be disabled)
            - Client Credentials Grant
            - Refresh Token Grant

        .. _`HTTP Basic Authentication Scheme`:
        https://tools.ietf.org/html/rfc1945#section-11.1
        """
    raise NotImplementedError('Subclasses must implement this method.')