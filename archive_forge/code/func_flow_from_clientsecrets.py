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
@_helpers.positional(2)
def flow_from_clientsecrets(filename, scope, redirect_uri=None, message=None, cache=None, login_hint=None, device_uri=None, pkce=None, code_verifier=None, prompt=None):
    """Create a Flow from a clientsecrets file.

    Will create the right kind of Flow based on the contents of the
    clientsecrets file or will raise InvalidClientSecretsError for unknown
    types of Flows.

    Args:
        filename: string, File name of client secrets.
        scope: string or iterable of strings, scope(s) to request.
        redirect_uri: string, Either the string 'urn:ietf:wg:oauth:2.0:oob' for
                      a non-web-based application, or a URI that handles the
                      callback from the authorization server.
        message: string, A friendly string to display to the user if the
                 clientsecrets file is missing or invalid. If message is
                 provided then sys.exit will be called in the case of an error.
                 If message in not provided then
                 clientsecrets.InvalidClientSecretsError will be raised.
        cache: An optional cache service client that implements get() and set()
               methods. See clientsecrets.loadfile() for details.
        login_hint: string, Either an email address or domain. Passing this
                    hint will either pre-fill the email box on the sign-in form
                    or select the proper multi-login session, thereby
                    simplifying the login flow.
        device_uri: string, URI for device authorization endpoint. For
                    convenience defaults to Google's endpoints but any
                    OAuth 2.0 provider can be used.

    Returns:
        A Flow object.

    Raises:
        UnknownClientSecretsFlowError: if the file describes an unknown kind of
                                       Flow.
        clientsecrets.InvalidClientSecretsError: if the clientsecrets file is
                                                 invalid.
    """
    try:
        client_type, client_info = clientsecrets.loadfile(filename, cache=cache)
        if client_type in (clientsecrets.TYPE_WEB, clientsecrets.TYPE_INSTALLED):
            constructor_kwargs = {'redirect_uri': redirect_uri, 'auth_uri': client_info['auth_uri'], 'token_uri': client_info['token_uri'], 'login_hint': login_hint}
            revoke_uri = client_info.get('revoke_uri')
            optional = ('revoke_uri', 'device_uri', 'pkce', 'code_verifier', 'prompt')
            for param in optional:
                if locals()[param] is not None:
                    constructor_kwargs[param] = locals()[param]
            return OAuth2WebServerFlow(client_info['client_id'], client_info['client_secret'], scope, **constructor_kwargs)
    except clientsecrets.InvalidClientSecretsError as e:
        if message is not None:
            if e.args:
                message = 'The client secrets were invalid: \n{0}\n{1}'.format(e, message)
            sys.exit(message)
        else:
            raise
    else:
        raise UnknownClientSecretsFlowError('This OAuth 2.0 flow is unsupported: {0!r}'.format(client_type))