import datetime
import errno
import json
import os
import requests
import sys
import time
import webbrowser
import google_auth_oauthlib.flow as auth_flows
import grpc
import google.auth
import google.auth.transport.requests
import google.oauth2.credentials
from tensorboard.uploader import util
from tensorboard.util import tb_logging
def authenticate_user(force_console=False) -> google.oauth2.credentials.Credentials:
    """Makes the user authenticate to retrieve auth credentials.

    The default behavior is to use the [installed app flow](
    http://developers.google.com/identity/protocols/oauth2/native-app), in which
    a browser is started for the user to authenticate, along with a local web
    server. The authentication in the browser would produce a redirect response
    to `localhost` with an authorization code that would then be received by the
    local web server started here.

    The two most notable cases where the default flow is not well supported are:
    - When the uploader is run from a colab notebook.
    - Then the uploader is run via a remote terminal (SSH).

    If any of the following is true, a different auth flow will be used:
    - the flag `--auth_force_console` is set to true, or
    - a browser is not available, or
    - a local web server cannot be started

    In this case, a [limited-input device flow](
    http://developers.google.com/identity/protocols/oauth2/limited-input-device)
    will be used, in which the user is presented with a URL and a short code
    that they'd need to use to authenticate and authorize access in a separate
    browser or device. The uploader will poll for access until the access is
    granted or rejected, or the initiated authorization request expires.
    """
    scopes = OPENID_CONNECT_SCOPES
    if not force_console and os.getenv('DISPLAY'):
        try:
            client_config = json.loads(_INSTALLED_APP_OAUTH_CLIENT_CONFIG)
            flow = auth_flows.InstalledAppFlow.from_client_config(client_config, scopes=scopes)
            return flow.run_local_server(port=0)
        except webbrowser.Error:
            sys.stderr.write('Falling back to remote authentication flow...\n')
    client_config = json.loads(_LIMITED_INPUT_DEVICE_OAUTH_CLIENT_CONFIG)
    flow = _LimitedInputDeviceAuthFlow(client_config, scopes=scopes)
    return flow.run()