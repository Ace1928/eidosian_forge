import unittest
import macaroonbakery.bakery as bakery
import macaroonbakery.checkers as checkers
from macaroonbakery.tests import common
from pymacaroons.verifier import Verifier
def test_discharge_all_local_discharge(self):
    oc = common.new_bakery('ts', None)
    client_key = bakery.generate_key()
    m = oc.oven.macaroon(bakery.LATEST_VERSION, common.ages, [bakery.local_third_party_caveat(client_key.public_key, bakery.LATEST_VERSION)], [bakery.LOGIN_OP])
    ms = bakery.discharge_all(m, no_discharge(self), client_key)
    oc.checker.auth([ms]).allow(common.test_context, [bakery.LOGIN_OP])