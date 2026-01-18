import urllib.parse as urlparse
def get_oauth_token_from_body(body):
    """Parse the URL response body to retrieve the oauth token key and secret.

    The response body will look like:
    'oauth_token=12345&oauth_token_secret=67890' with
    'oauth_expires_at=2013-03-30T05:27:19.463201' possibly there, too.
    """
    body = body.decode('utf-8')
    credentials = urlparse.parse_qs(body)
    key = credentials['oauth_token'][0]
    secret = credentials['oauth_token_secret'][0]
    token = {'key': key, 'id': key, 'secret': secret}
    expires_at = credentials.get('oauth_expires_at')
    if expires_at:
        token['expires'] = expires_at[0]
    return token