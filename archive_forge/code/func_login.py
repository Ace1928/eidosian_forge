import json
import logging
import os
import tempfile
from datetime import datetime, timedelta
from unittest import TestCase
import macaroonbakery.bakery as bakery
import macaroonbakery.checkers as checkers
import macaroonbakery.httpbakery as httpbakery
import macaroonbakery.httpbakery.agent as agent
import requests.cookies
from httmock import HTTMock, response, urlmatch
from six.moves.urllib.parse import parse_qs, urlparse
@urlmatch(path='.*/login')
def login(url, request):
    qs = parse_qs(urlparse(request.url).query)
    self.assertEqual(request.method, 'GET')
    self.assertEqual(qs, {'username': ['test-user'], 'public-key': [PUBLIC_KEY]})
    b = bakery.Bakery(key=discharge_key)
    m = b.oven.macaroon(version=bakery.LATEST_VERSION, expiry=datetime.utcnow() + timedelta(days=1), caveats=[bakery.local_third_party_caveat(PUBLIC_KEY, version=httpbakery.request_version(request.headers))], ops=[bakery.Op(entity='agent', action='login')])
    return {'status_code': 200, 'content': {'macaroon': m.to_dict()}}