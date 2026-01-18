import collections
import copy
import datetime
import json
import logging
import os
import shutil
import socket
import sys
import tempfile
import six
from six.moves import http_client
from six.moves import urllib
import oauth2client
from oauth2client import _helpers
from oauth2client import _pkce
from oauth2client import clientsecrets
from oauth2client import transport
@_helpers.positional(4)
def credentials_from_code(client_id, client_secret, scope, code, redirect_uri='postmessage', http=None, user_agent=None, token_uri=oauth2client.GOOGLE_TOKEN_URI, auth_uri=oauth2client.GOOGLE_AUTH_URI, revoke_uri=oauth2client.GOOGLE_REVOKE_URI, device_uri=oauth2client.GOOGLE_DEVICE_URI, token_info_uri=oauth2client.GOOGLE_TOKEN_INFO_URI, pkce=False, code_verifier=None):
    """Exchanges an authorization code for an OAuth2Credentials object.

    Args:
        client_id: string, client identifier.
        client_secret: string, client secret.
        scope: string or iterable of strings, scope(s) to request.
        code: string, An authorization code, most likely passed down from
              the client
        redirect_uri: string, this is generally set to 'postmessage' to match
                      the redirect_uri that the client specified
        http: httplib2.Http, optional http instance to use to do the fetch
        token_uri: string, URI for token endpoint. For convenience defaults
                   to Google's endpoints but any OAuth 2.0 provider can be
                   used.
        auth_uri: string, URI for authorization endpoint. For convenience
                  defaults to Google's endpoints but any OAuth 2.0 provider
                  can be used.
        revoke_uri: string, URI for revoke endpoint. For convenience
                    defaults to Google's endpoints but any OAuth 2.0 provider
                    can be used.
        device_uri: string, URI for device authorization endpoint. For
                    convenience defaults to Google's endpoints but any OAuth
                    2.0 provider can be used.
        pkce: boolean, default: False, Generate and include a "Proof Key
              for Code Exchange" (PKCE) with your authorization and token
              requests. This adds security for installed applications that
              cannot protect a client_secret. See RFC 7636 for details.
        code_verifier: bytestring or None, default: None, parameter passed
                       as part of the code exchange when pkce=True. If
                       None, a code_verifier will automatically be
                       generated as part of step1_get_authorize_url(). See
                       RFC 7636 for details.

    Returns:
        An OAuth2Credentials object.

    Raises:
        FlowExchangeError if the authorization code cannot be exchanged for an
        access token
    """
    flow = OAuth2WebServerFlow(client_id, client_secret, scope, redirect_uri=redirect_uri, user_agent=user_agent, auth_uri=auth_uri, token_uri=token_uri, revoke_uri=revoke_uri, device_uri=device_uri, token_info_uri=token_info_uri, pkce=pkce, code_verifier=code_verifier)
    credentials = flow.step2_exchange(code, http=http)
    return credentials