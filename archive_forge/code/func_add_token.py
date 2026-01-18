from __future__ import absolute_import, unicode_literals
import time
import warnings
from oauthlib.common import generate_token
from oauthlib.oauth2.rfc6749 import tokens
from oauthlib.oauth2.rfc6749.errors import (InsecureTransportError,
from oauthlib.oauth2.rfc6749.parameters import (parse_token_response,
from oauthlib.oauth2.rfc6749.utils import is_secure_transport
def add_token(self, uri, http_method='GET', body=None, headers=None, token_placement=None, **kwargs):
    """Add token to the request uri, body or authorization header.

        The access token type provides the client with the information
        required to successfully utilize the access token to make a protected
        resource request (along with type-specific attributes).  The client
        MUST NOT use an access token if it does not understand the token
        type.

        For example, the "bearer" token type defined in
        [`I-D.ietf-oauth-v2-bearer`_] is utilized by simply including the access
        token string in the request:

        .. code-block:: http

            GET /resource/1 HTTP/1.1
            Host: example.com
            Authorization: Bearer mF_9.B5f-4.1JqM

        while the "mac" token type defined in [`I-D.ietf-oauth-v2-http-mac`_] is
        utilized by issuing a MAC key together with the access token which is
        used to sign certain components of the HTTP requests:

        .. code-block:: http

            GET /resource/1 HTTP/1.1
            Host: example.com
            Authorization: MAC id="h480djs93hd8",
                                nonce="274312:dj83hs9s",
                                mac="kDZvddkndxvhGRXZhvuDjEWhGeE="

        .. _`I-D.ietf-oauth-v2-bearer`:
        https://tools.ietf.org/html/rfc6749#section-12.2
        .. _`I-D.ietf-oauth-v2-http-mac`:
        https://tools.ietf.org/html/rfc6749#section-12.2
        """
    if not is_secure_transport(uri):
        raise InsecureTransportError()
    token_placement = token_placement or self.default_token_placement
    case_insensitive_token_types = dict(((k.lower(), v) for k, v in self.token_types.items()))
    if not self.token_type.lower() in case_insensitive_token_types:
        raise ValueError('Unsupported token type: %s' % self.token_type)
    if not (self.access_token or self.token.get('access_token')):
        raise ValueError('Missing access token.')
    if self._expires_at and self._expires_at < time.time():
        raise TokenExpiredError()
    return case_insensitive_token_types[self.token_type.lower()](uri, http_method, body, headers, token_placement, **kwargs)