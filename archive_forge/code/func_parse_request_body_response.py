from __future__ import absolute_import, unicode_literals
import time
import warnings
from oauthlib.common import generate_token
from oauthlib.oauth2.rfc6749 import tokens
from oauthlib.oauth2.rfc6749.errors import (InsecureTransportError,
from oauthlib.oauth2.rfc6749.parameters import (parse_token_response,
from oauthlib.oauth2.rfc6749.utils import is_secure_transport
def parse_request_body_response(self, body, scope=None, **kwargs):
    """Parse the JSON response body.

        If the access token request is valid and authorized, the
        authorization server issues an access token as described in
        `Section 5.1`_.  A refresh token SHOULD NOT be included.  If the request
        failed client authentication or is invalid, the authorization server
        returns an error response as described in `Section 5.2`_.

        :param body: The response body from the token request.
        :param scope: Scopes originally requested.
        :return: Dictionary of token parameters.
        :raises: Warning if scope has changed. OAuth2Error if response is
        invalid.

        These response are json encoded and could easily be parsed without
        the assistance of OAuthLib. However, there are a few subtle issues
        to be aware of regarding the response which are helpfully addressed
        through the raising of various errors.

        A successful response should always contain

        **access_token**
                The access token issued by the authorization server. Often
                a random string.

        **token_type**
            The type of the token issued as described in `Section 7.1`_.
            Commonly ``Bearer``.

        While it is not mandated it is recommended that the provider include

        **expires_in**
            The lifetime in seconds of the access token.  For
            example, the value "3600" denotes that the access token will
            expire in one hour from the time the response was generated.
            If omitted, the authorization server SHOULD provide the
            expiration time via other means or document the default value.

        **scope**
            Providers may supply this in all responses but are required to only
            if it has changed since the authorization request.

        .. _`Section 5.1`: https://tools.ietf.org/html/rfc6749#section-5.1
        .. _`Section 5.2`: https://tools.ietf.org/html/rfc6749#section-5.2
        .. _`Section 7.1`: https://tools.ietf.org/html/rfc6749#section-7.1
        """
    self.token = parse_token_response(body, scope=scope)
    self.populate_token_attributes(self.token)
    return self.token