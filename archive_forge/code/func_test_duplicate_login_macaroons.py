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
def test_duplicate_login_macaroons(self):
    locator = _DischargerLocator()
    ids = _IdService('ids', locator, self)
    auth = bakery.ClosedAuthorizer()
    ts = _Service('myservice', auth, ids, locator)
    client1 = _Client(locator)
    ctx = test_context.with_value(_DISCHARGE_USER_KEY, 'bob')
    auth_info = client1.do(ctx, ts, [bakery.LOGIN_OP])
    self.assertEqual(auth_info.identity.id(), 'bob')
    client2 = _Client(locator)
    ctx = test_context.with_value(_DISCHARGE_USER_KEY, 'alice')
    auth_info = client2.do(ctx, ts, [bakery.LOGIN_OP])
    self.assertEqual(auth_info.identity.id(), 'alice')
    client3 = _Client(locator)
    client3.add_macaroon(ts, '1.bob', client1._macaroons[ts.name()]['authn'])
    client3.add_macaroon(ts, '2.alice', client2._macaroons[ts.name()]['authn'])
    auth_info = client3.do(test_context, ts, [bakery.LOGIN_OP])
    self.assertEqual(auth_info.identity.id(), 'bob')
    self.assertEqual(len(auth_info.macaroons), 1)
    client3 = _Client(locator)
    client3.add_macaroon(ts, '1.alice', client2._macaroons[ts.name()]['authn'])
    client3.add_macaroon(ts, '2.bob', client1._macaroons[ts.name()]['authn'])
    auth_info = client3.do(test_context, ts, [bakery.LOGIN_OP])
    self.assertEqual(auth_info.identity.id(), 'alice')
    self.assertEqual(len(auth_info.macaroons), 1)