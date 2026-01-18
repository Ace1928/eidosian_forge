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
def test_macaroon_ops_fatal_error(self):
    checker = bakery.Checker(macaroon_opstore=_MacaroonStoreWithError())
    m = pymacaroons.Macaroon(version=pymacaroons.MACAROON_V2)
    with self.assertRaises(bakery.AuthInitError):
        checker.auth([m]).allow(test_context, [bakery.LOGIN_OP])