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
def id_token_call_credentials(credentials):
    """Constructs `grpc.CallCredentials` using
    `google.auth.Credentials.id_token`.

    Args:
      credentials (google.auth.credentials.Credentials): The credentials to use.

    Returns:
      grpc.CallCredentials: The call credentials.
    """
    request = google.auth.transport.requests.Request()
    return grpc.metadata_call_credentials(IdTokenAuthMetadataPlugin(credentials, request))