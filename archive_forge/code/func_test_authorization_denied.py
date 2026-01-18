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
def test_authorization_denied(self):
    locator = _DischargerLocator()
    ids = _IdService('ids', locator, self)
    auth = bakery.ClosedAuthorizer()
    ts = _Service('myservice', auth, ids, locator)
    client = _Client(locator)
    ctx = test_context.with_value(_DISCHARGE_USER_KEY, 'bob')
    with self.assertRaises(bakery.PermissionDenied):
        client.do(ctx, ts, [bakery.Op(entity='something', action='read')])