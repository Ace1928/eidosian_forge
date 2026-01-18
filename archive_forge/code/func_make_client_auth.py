import json
import pytest  # type: ignore
from google.auth import exceptions
from google.oauth2 import utils
@classmethod
def make_client_auth(cls, client_secret=None):
    return utils.ClientAuthentication(utils.ClientAuthType.basic, CLIENT_ID, client_secret)