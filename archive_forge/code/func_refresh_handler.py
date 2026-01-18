import re
import uuid
import google.auth
from google.auth import downscoped
from google.auth.transport import requests
from google.cloud import exceptions
from google.cloud import storage
from google.oauth2 import credentials
import pytest
def refresh_handler(request, scopes=None):
    return get_token_from_broker(bucket_name, _OBJECT_PREFIX)