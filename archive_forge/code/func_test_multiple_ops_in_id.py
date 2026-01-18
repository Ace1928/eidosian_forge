import copy
from datetime import datetime, timedelta
from unittest import TestCase
import macaroonbakery.bakery as bakery
def test_multiple_ops_in_id(self):
    test_oven = bakery.Oven()
    ops = [bakery.Op('one', 'read'), bakery.Op('one', 'write'), bakery.Op('two', 'read')]
    m = test_oven.macaroon(bakery.LATEST_VERSION, AGES, None, ops)
    got_ops, conds = test_oven.macaroon_ops([m.macaroon])
    self.assertEqual(len(conds), 1)
    self.assertEqual(bakery.canonical_ops(got_ops), ops)