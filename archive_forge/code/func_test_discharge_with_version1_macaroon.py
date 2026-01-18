import os
import unittest
import macaroonbakery.bakery as bakery
import macaroonbakery.checkers as checkers
from macaroonbakery.tests import common
from pymacaroons import MACAROON_V1, Macaroon
def test_discharge_with_version1_macaroon(self):
    locator = bakery.ThirdPartyStore()
    bs = common.new_bakery('bs-loc', locator)
    ts = common.new_bakery('ts-loc', locator)
    ts_macaroon = ts.oven.macaroon(bakery.VERSION_1, common.ages, None, [bakery.LOGIN_OP])
    ts_macaroon.add_caveat(checkers.Caveat(condition='something', location='bs-loc'), ts.oven.key, ts.oven.locator)

    def get_discharge(cav, payload):
        try:
            cav.caveat_id_bytes.decode('utf-8')
        except UnicodeDecodeError:
            self.fail('caveat id is not utf-8')
        return bakery.discharge(common.test_context, cav.caveat_id_bytes, payload, bs.oven.key, common.ThirdPartyStrcmpChecker('something'), bs.oven.locator)
    d = bakery.discharge_all(ts_macaroon, get_discharge)
    ts.checker.auth([d]).allow(common.test_context, [bakery.LOGIN_OP])
    for m in d:
        self.assertEqual(m.version, MACAROON_V1)