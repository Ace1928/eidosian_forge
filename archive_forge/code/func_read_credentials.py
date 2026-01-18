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
def read_credentials(self):
    """Returns the current `google.oauth2.credentials.Credentials`, or
        None."""
    if self._credentials_filepath is None:
        return None
    if os.path.exists(self._credentials_filepath):
        return google.oauth2.credentials.Credentials.from_authorized_user_file(self._credentials_filepath)
    return None