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
def test_authorize_with_authentication_required(self):
    locator = _DischargerLocator()
    ids = _IdService('ids', locator, self)
    auth = _OpAuthorizer({bakery.Op(entity='something', action='read'): {'bob'}})
    ts = _Service('myservice', auth, ids, locator)
    client = _Client(locator)
    ctx = test_context.with_value(_DISCHARGE_USER_KEY, 'bob')
    auth_info = client.do(ctx, ts, [bakery.Op(entity='something', action='read')])
    self.assertEqual(self._discharges, [_DischargeRecord(location='ids', user='bob')])
    self.assertIsNotNone(auth_info)
    self.assertEqual(auth_info.identity.id(), 'bob')
    self.assertEqual(len(auth_info.macaroons), 1)