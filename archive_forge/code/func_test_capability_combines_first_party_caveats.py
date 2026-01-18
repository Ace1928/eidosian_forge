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
def test_capability_combines_first_party_caveats(self):
    locator = _DischargerLocator()
    ids = _IdService('ids', locator, self)
    auth = _OpAuthorizer({bakery.Op(entity='e1', action='read'): {'alice'}, bakery.Op(entity='e2', action='read'): {'bob'}})
    ts = _Service('myservice', auth, ids, locator)
    ctx = test_context.with_value(_DISCHARGE_USER_KEY, 'alice')
    m1 = _Client(locator).capability(ctx, ts, [bakery.Op(entity='e1', action='read')])
    m1.macaroon.add_first_party_caveat('true 1')
    m1.macaroon.add_first_party_caveat('true 2')
    ctx = test_context.with_value(_DISCHARGE_USER_KEY, 'bob')
    m2 = _Client(locator).capability(ctx, ts, [bakery.Op(entity='e2', action='read')])
    m2.macaroon.add_first_party_caveat('true 3')
    m2.macaroon.add_first_party_caveat('true 4')
    client = _Client(locator)
    client.add_macaroon(ts, 'authz1', [m1.macaroon])
    client.add_macaroon(ts, 'authz2', [m2.macaroon])
    m = client.capability(test_context, ts, [bakery.Op(entity='e1', action='read'), bakery.Op(entity='e2', action='read')])
    self.assertEqual(_macaroon_conditions(m.macaroon.caveats, False), ['true 1', 'true 2', 'true 3', 'true 4'])