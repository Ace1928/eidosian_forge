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
def authorization_url(self, **kwargs):
    """Generates an authorization URL.

        This is the first step in the OAuth 2.0 Authorization Flow. The user's
        browser should be redirected to the returned URL.

        This method calls
        :meth:`requests_oauthlib.OAuth2Session.authorization_url`
        and specifies the client configuration's authorization URI (usually
        Google's authorization server) and specifies that "offline" access is
        desired. This is required in order to obtain a refresh token.

        Args:
            kwargs: Additional arguments passed through to
                :meth:`requests_oauthlib.OAuth2Session.authorization_url`

        Returns:
            Tuple[str, str]: The generated authorization URL and state. The
                user must visit the URL to complete the flow. The state is used
                when completing the flow to verify that the request originated
                from your application. If your application is using a different
                :class:`Flow` instance to obtain the token, you will need to
                specify the ``state`` when constructing the :class:`Flow`.
        """
    kwargs.setdefault('access_type', 'offline')
    if self.autogenerate_code_verifier:
        chars = ascii_letters + digits + '-._~'
        rnd = SystemRandom()
        random_verifier = [rnd.choice(chars) for _ in range(0, 128)]
        self.code_verifier = ''.join(random_verifier)
    if self.code_verifier:
        code_hash = hashlib.sha256()
        code_hash.update(str.encode(self.code_verifier))
        unencoded_challenge = code_hash.digest()
        b64_challenge = urlsafe_b64encode(unencoded_challenge)
        code_challenge = b64_challenge.decode().split('=')[0]
        kwargs.setdefault('code_challenge', code_challenge)
        kwargs.setdefault('code_challenge_method', 'S256')
    url, state = self.oauth2session.authorization_url(self.client_config['auth_uri'], **kwargs)
    return (url, state)