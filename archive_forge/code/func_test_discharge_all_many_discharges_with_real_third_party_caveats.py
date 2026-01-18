import unittest
import macaroonbakery.bakery as bakery
import macaroonbakery.checkers as checkers
from macaroonbakery.tests import common
from pymacaroons.verifier import Verifier
def test_discharge_all_many_discharges_with_real_third_party_caveats(self):
    locator = bakery.ThirdPartyStore()
    bakeries = {}
    total_discharges_required = 40

    class M:
        bakery_id = 0
        still_required = total_discharges_required

    def add_bakery():
        M.bakery_id += 1
        loc = 'loc{}'.format(M.bakery_id)
        bakeries[loc] = common.new_bakery(loc, locator)
        return loc
    ts = common.new_bakery('ts-loc', locator)

    def checker(_, ci):
        caveats = []
        if ci.condition != 'something':
            self.fail('unexpected condition')
        for i in range(0, 2):
            if M.still_required <= 0:
                break
            caveats.append(checkers.Caveat(location=add_bakery(), condition='something'))
            M.still_required -= 1
        return caveats
    root_key = b'root key'
    m0 = bakery.Macaroon(root_key=root_key, id=b'id0', location='ts-loc', version=bakery.LATEST_VERSION)
    m0.add_caveat(checkers.Caveat(location=add_bakery(), condition='something'), ts.oven.key, locator)
    M.still_required -= 1

    class ThirdPartyCaveatCheckerF(bakery.ThirdPartyCaveatChecker):

        def check_third_party_caveat(self, ctx, info):
            return checker(ctx, info)

    def get_discharge(cav, payload):
        return bakery.discharge(common.test_context, cav.caveat_id, payload, bakeries[cav.location].oven.key, ThirdPartyCaveatCheckerF(), locator)
    ms = bakery.discharge_all(m0, get_discharge)
    self.assertEqual(len(ms), total_discharges_required + 1)
    v = Verifier()
    v.satisfy_general(always_ok)
    v.verify(ms[0], root_key, ms[1:])