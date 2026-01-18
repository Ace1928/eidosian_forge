import io
import json
import logging
import os
import warnings
import six
from google.auth import environment_vars
from google.auth import exceptions
import google.auth.transport._http_client
def get_api_key_credentials(key):
    """Return credentials with the given API key."""
    from google.auth import api_key
    return api_key.Credentials(key)