import json
import os
from six.moves import http_client
import oauth2client
from oauth2client import client
from oauth2client import service_account
from oauth2client import transport
def run_p12():
    credentials = service_account.ServiceAccountCredentials.from_p12_keyfile(P12_KEY_EMAIL, P12_KEY_PATH, scopes=SCOPE)
    _check_user_info(credentials, P12_KEY_EMAIL)