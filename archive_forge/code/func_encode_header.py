import base64
import json
import logging
from . import credentials
from . import errors
from .utils import config
def encode_header(auth):
    auth_json = json.dumps(auth).encode('ascii')
    return base64.urlsafe_b64encode(auth_json)