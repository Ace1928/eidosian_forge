import base64
import binascii
import ipaddress
import json
import webbrowser
from datetime import datetime
import six
from pymacaroons import Macaroon
from pymacaroons.serializers import json_serializer
import six.moves.http_cookiejar as http_cookiejar
from six.moves.urllib.parse import urlparse
def raw_urlsafe_b64encode(b):
    """Base64 encode using URL-safe encoding with padding removed.

    @param b bytes to decode
    @return bytes decoded
    """
    b = to_bytes(b)
    b = base64.urlsafe_b64encode(b)
    b = b.rstrip(b'=')
    return b