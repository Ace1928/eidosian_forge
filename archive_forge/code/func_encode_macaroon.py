import base64
import datetime
import json
import platform
import threading
import macaroonbakery.bakery as bakery
import macaroonbakery.checkers as checkers
import macaroonbakery.httpbakery as httpbakery
import pymacaroons
import requests
import macaroonbakery._utils as utils
from macaroonbakery.httpbakery._error import DischargeError
from fixtures import (
from httmock import HTTMock, urlmatch
from six.moves.urllib.parse import parse_qs
from six.moves.urllib.request import Request
def encode_macaroon(m):
    macaroons = '[' + utils.macaroon_to_json_string(m) + ']'
    return base64.urlsafe_b64encode(utils.to_bytes(macaroons)).decode('ascii')