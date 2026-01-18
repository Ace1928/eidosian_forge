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
def test_auth_with_third_party_caveats(self):
    locator = _DischargerLocator()
    ids = _IdService('ids', locator, self)

    def authorize_with_tp_discharge(ctx, id, op):
        if id is not None and id.id() == 'bob' and (op == bakery.Op(entity='something', action='read')):
            return (True, [checkers.Caveat(condition='question', location='other third party')])
        return (False, None)
    auth = bakery.AuthorizerFunc(authorize_with_tp_discharge)
    ts = _Service('myservice', auth, ids, locator)

    class _LocalDischargeChecker(bakery.ThirdPartyCaveatChecker):

        def check_third_party_caveat(_, ctx, info):
            if info.condition != 'question':
                raise ValueError('third party condition not recognized')
            self._discharges.append(_DischargeRecord(location='other third party', user=ctx.get(_DISCHARGE_USER_KEY)))
            return []
    locator['other third party'] = _Discharger(key=bakery.generate_key(), checker=_LocalDischargeChecker(), locator=locator)
    client = _Client(locator)
    ctx = test_context.with_value(_DISCHARGE_USER_KEY, 'bob')
    client.do(ctx, ts, [bakery.Op(entity='something', action='read')])
    self.assertEqual(self._discharges, [_DischargeRecord(location='ids', user='bob'), _DischargeRecord(location='other third party', user='bob')])