import binascii
import hashlib
import hmac
import ipaddress
import logging
import urllib.parse as urlparse
import warnings
from oauthlib.common import extract_params, safe_string_equals, urldecode
from . import utils
def sign_hmac_sha512_with_client(sig_base_str: str, client):
    return _sign_hmac('SHA-512', sig_base_str, client.client_secret, client.resource_owner_secret)