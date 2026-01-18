import json
import os
from six.moves import http_client
import oauth2client
from oauth2client import client
from oauth2client import service_account
from oauth2client import transport
def run_json():
    factory = service_account.ServiceAccountCredentials.from_json_keyfile_name
    credentials = factory(JSON_KEY_PATH, scopes=SCOPE)
    service_account_email = credentials._service_account_email
    _check_user_info(credentials, service_account_email)