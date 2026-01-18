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
def test_expiry_cookie_set_in_past(self):

    class _DischargerLocator(bakery.ThirdPartyLocator):

        def __init__(self):
            self.key = bakery.generate_key()

        def third_party_info(self, loc):
            if loc == 'http://1.2.3.4':
                return bakery.ThirdPartyInfo(public_key=self.key.public_key, version=bakery.LATEST_VERSION)
    d = _DischargerLocator()
    b = new_bakery('loc', d, None)

    @urlmatch(path='.*/discharge')
    def discharge(url, request):
        qs = parse_qs(request.body)
        content = {q: qs[q][0] for q in qs}
        m = httpbakery.discharge(checkers.AuthContext(), content, d.key, d, alwaysOK3rd)
        return {'status_code': 200, 'content': {'Macaroon': m.to_dict()}}
    ages = datetime.datetime.utcnow() - datetime.timedelta(days=1)

    def handler(*args):
        GetHandler(b, 'http://1.2.3.4', None, None, None, ages, *args)
    try:
        httpd = HTTPServer(('', 0), handler)
        thread = threading.Thread(target=httpd.serve_forever)
        thread.start()
        client = httpbakery.Client()
        with HTTMock(discharge):
            with self.assertRaises(httpbakery.BakeryException) as ctx:
                requests.get(url='http://' + httpd.server_address[0] + ':' + str(httpd.server_address[1]), cookies=client.cookies, auth=client.auth())
        self.assertEqual(ctx.exception.args[0], 'too many (3) discharge requests')
    finally:
        httpd.shutdown()