from base64 import urlsafe_b64encode
import hashlib
import json
import logging
from string import ascii_letters, digits
import webbrowser
import wsgiref.simple_server
import wsgiref.util
import google.auth.transport.requests
import google.oauth2.credentials
import google_auth_oauthlib.helpers
def fetch_token(self, **kwargs):
    """Completes the Authorization Flow and obtains an access token.

        This is the final step in the OAuth 2.0 Authorization Flow. This is
        called after the user consents.

        This method calls
        :meth:`requests_oauthlib.OAuth2Session.fetch_token`
        and specifies the client configuration's token URI (usually Google's
        token server).

        Args:
            kwargs: Arguments passed through to
                :meth:`requests_oauthlib.OAuth2Session.fetch_token`. At least
                one of ``code`` or ``authorization_response`` must be
                specified.

        Returns:
            Mapping[str, str]: The obtained tokens. Typically, you will not use
                return value of this function and instead and use
                :meth:`credentials` to obtain a
                :class:`~google.auth.credentials.Credentials` instance.
        """
    kwargs.setdefault('client_secret', self.client_config['client_secret'])
    kwargs.setdefault('code_verifier', self.code_verifier)
    return self.oauth2session.fetch_token(self.client_config['token_uri'], **kwargs)