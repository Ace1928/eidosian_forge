from __future__ import print_function
import httplib2
import json
import os
from select import select
import stat
from sys import stdin
import time
import webbrowser
from base64 import (
from six.moves.urllib.parse import parse_qs
from lazr.restfulclient.errors import HTTPError
from lazr.restfulclient.authorize.oauth import (
from launchpadlib import uris
def exchange_request_token_for_access_token(self, web_root=uris.STAGING_WEB_ROOT):
    """Exchange the previously obtained request token for an access token.

        This method must not be called unless get_request_token() has been
        called and completed successfully.

        The access token will be stored as self.access_token.

        :param web_root: The base URL of the website that granted the
            request token.
        """
    assert self._request_token is not None, "get_request_token() doesn't seem to have been called."
    web_root = uris.lookup_web_root(web_root)
    params = dict(oauth_consumer_key=self.consumer.key, oauth_signature_method='PLAINTEXT', oauth_token=self._request_token.key, oauth_signature='&%s' % self._request_token.secret)
    url = web_root + access_token_page
    headers = {'Referer': web_root}
    response, content = _http_post(url, headers, params)
    self.access_token = AccessToken.from_string(content)