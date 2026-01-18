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
def test_extract_macaroons_from_request(self):

    def encode_macaroon(m):
        macaroons = '[' + utils.macaroon_to_json_string(m) + ']'
        return base64.urlsafe_b64encode(utils.to_bytes(macaroons)).decode('ascii')
    req = Request('http://example.com')
    m1 = pymacaroons.Macaroon(version=pymacaroons.MACAROON_V2, identifier='one')
    req.add_header('Macaroons', encode_macaroon(m1))
    m2 = pymacaroons.Macaroon(version=pymacaroons.MACAROON_V2, identifier='two')
    jar = requests.cookies.RequestsCookieJar()
    jar.set_cookie(utils.cookie(name='macaroon-auth', value=encode_macaroon(m2), url='http://example.com'))
    jar.set_cookie(utils.cookie(name='macaroon-empty', value='', url='http://example.com'))
    jar.add_cookie_header(req)
    macaroons = httpbakery.extract_macaroons(req)
    self.assertEqual(len(macaroons), 2)
    macaroons.sort(key=lambda ms: ms[0].identifier)
    self.assertEqual(macaroons[0][0].identifier, m1.identifier)
    self.assertEqual(macaroons[1][0].identifier, m2.identifier)