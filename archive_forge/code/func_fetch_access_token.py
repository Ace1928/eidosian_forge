from urllib.parse import urlparse
import logging
from oauthlib.common import add_params_to_uri
from oauthlib.common import urldecode as _urldecode
from oauthlib.oauth1 import SIGNATURE_HMAC, SIGNATURE_RSA, SIGNATURE_TYPE_AUTH_HEADER
import requests
from . import OAuth1
def fetch_access_token(self, url, verifier=None, **request_kwargs):
    """Fetch an access token.

        This is the final step in the OAuth 1 workflow. An access token is
        obtained using all previously obtained credentials, including the
        verifier from the authorization step.

        Note that a previously set verifier will be reset for your
        convenience, or else signature creation will be incorrect on
        consecutive requests.

        >>> access_token_url = 'https://api.twitter.com/oauth/access_token'
        >>> redirect_response = 'https://127.0.0.1/callback?oauth_token=kjerht2309uf&oauth_token_secret=lsdajfh923874&oauth_verifier=w34o8967345'
        >>> oauth_session = OAuth1Session('client-key', client_secret='secret')
        >>> oauth_session.parse_authorization_response(redirect_response)
        {
            'oauth_token: 'kjerht2309u',
            'oauth_token_secret: 'lsdajfh923874',
            'oauth_verifier: 'w34o8967345',
        }
        >>> oauth_session.fetch_access_token(access_token_url)
        {
            'oauth_token': 'sdf0o9823sjdfsdf',
            'oauth_token_secret': '2kjshdfp92i34asdasd',
        }
        """
    if verifier:
        self._client.client.verifier = verifier
    if not getattr(self._client.client, 'verifier', None):
        raise VerifierMissing('No client verifier has been set.')
    token = self._fetch_token(url, **request_kwargs)
    log.debug('Resetting verifier attribute, should not be used anymore.')
    self._client.client.verifier = None
    return token