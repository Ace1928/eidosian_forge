import base64
import json
from collections import namedtuple
from datetime import timedelta
from unittest import TestCase
import macaroonbakery.bakery as bakery
import macaroonbakery.checkers as checkers
import pymacaroons
from macaroonbakery.tests.common import epoch, test_checker, test_context
from pymacaroons.verifier import FirstPartyCaveatVerifierDelegate, Verifier
def test_capability(self):
    locator = _DischargerLocator()
    ids = _IdService('ids', locator, self)
    auth = _OpAuthorizer({bakery.Op(entity='something', action='read'): {'bob'}})
    ts = _Service('myservice', auth, ids, locator)
    client = _Client(locator)
    ctx = test_context.with_value(_DISCHARGE_USER_KEY, 'bob')
    m = client.discharged_capability(ctx, ts, [bakery.Op(entity='something', action='read')])
    auth_info = ts.do(test_context, [m], [bakery.Op(entity='something', action='read')])
    self.assertIsNotNone(auth_info)
    self.assertIsNone(auth_info.identity)
    self.assertEqual(len(auth_info.macaroons), 1)
    self.assertEqual(auth_info.macaroons[0][0].identifier_bytes, m[0].identifier_bytes)