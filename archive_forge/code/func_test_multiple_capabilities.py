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
def test_multiple_capabilities(self):
    locator = _DischargerLocator()
    ids = _IdService('ids', locator, self)
    auth = _OpAuthorizer({bakery.Op(entity='e1', action='read'): {'alice'}, bakery.Op(entity='e2', action='read'): {'bob'}})
    ts = _Service('myservice', auth, ids, locator)
    ctx = test_context.with_value(_DISCHARGE_USER_KEY, 'alice')
    m1 = _Client(locator).discharged_capability(ctx, ts, [bakery.Op(entity='e1', action='read')])
    ctx = test_context.with_value(_DISCHARGE_USER_KEY, 'bob')
    m2 = _Client(locator).discharged_capability(ctx, ts, [bakery.Op(entity='e2', action='read')])
    self.assertEqual(self._discharges, [_DischargeRecord(location='ids', user='alice'), _DischargeRecord(location='ids', user='bob')])
    auth_info = ts.do(test_context, [m1, m2], [bakery.Op(entity='e1', action='read'), bakery.Op(entity='e2', action='read')])
    self.assertIsNotNone(auth_info)
    self.assertIsNone(auth_info.identity)
    self.assertEqual(len(auth_info.macaroons), 2)
    self.assertEqual(auth_info.macaroons[0][0].identifier_bytes, m1[0].identifier_bytes)
    self.assertEqual(auth_info.macaroons[1][0].identifier_bytes, m2[0].identifier_bytes)