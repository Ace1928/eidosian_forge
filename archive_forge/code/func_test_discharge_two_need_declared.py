import os
import unittest
import macaroonbakery.bakery as bakery
import macaroonbakery.checkers as checkers
from macaroonbakery.tests import common
from pymacaroons import MACAROON_V1, Macaroon
def test_discharge_two_need_declared(self):
    locator = bakery.ThirdPartyStore()
    first_party = common.new_bakery('first', locator)
    third_party = common.new_bakery('third', locator)
    m = first_party.oven.macaroon(bakery.LATEST_VERSION, common.ages, [checkers.need_declared_caveat(checkers.Caveat(location='third', condition='x'), ['foo', 'bar']), checkers.need_declared_caveat(checkers.Caveat(location='third', condition='y'), ['bar', 'baz'])], [bakery.LOGIN_OP])

    def get_discharge(cav, payload):
        return bakery.discharge(common.test_context, cav.caveat_id_bytes, payload, third_party.oven.key, common.ThirdPartyCaveatCheckerEmpty(), third_party.oven.locator)
    d = bakery.discharge_all(m, get_discharge)
    declared = checkers.infer_declared(d, first_party.checker.namespace())
    self.assertEqual(declared, {'foo': '', 'bar': '', 'baz': ''})
    ctx = checkers.context_with_declared(common.test_context, declared)
    first_party.checker.auth([d]).allow(ctx, [bakery.LOGIN_OP])

    class ThirdPartyCaveatCheckerF(bakery.ThirdPartyCaveatChecker):

        def check_third_party_caveat(self, ctx, cav_info):
            if cav_info.condition == 'x':
                return [checkers.declared_caveat('foo', 'fooval1')]
            if cav_info.condition == 'y':
                return [checkers.declared_caveat('foo', 'fooval2'), checkers.declared_caveat('baz', 'bazval')]
            raise bakery.ThirdPartyCaveatCheckFailed('not matched')

    def get_discharge(cav, payload):
        return bakery.discharge(common.test_context, cav.caveat_id_bytes, payload, third_party.oven.key, ThirdPartyCaveatCheckerF(), third_party.oven.locator)
    d = bakery.discharge_all(m, get_discharge)
    declared = checkers.infer_declared(d, first_party.checker.namespace())
    self.assertEqual(declared, {'bar': '', 'baz': 'bazval'})
    with self.assertRaises(bakery.PermissionDenied) as exc:
        first_party.checker.auth([d]).allow(common.test_context, bakery.LOGIN_OP)
    self.assertEqual('cannot authorize login macaroon: caveat "declared foo fooval1" not satisfied: got foo=null, expected "fooval1"', exc.exception.args[0])