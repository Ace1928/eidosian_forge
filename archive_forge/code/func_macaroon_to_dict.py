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
def macaroon_to_dict(macaroon):
    """Turn macaroon into JSON-serializable dict object
    @param pymacaroons.Macaroon.
    """
    return json.loads(macaroon.serialize(json_serializer.JsonSerializer()))