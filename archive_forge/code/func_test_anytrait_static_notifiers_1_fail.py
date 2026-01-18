import unittest
from traits import trait_notifiers
from traits.api import Float, HasTraits, Undefined
def test_anytrait_static_notifiers_1_fail(self):
    obj = AnytraitStaticNotifiers1Fail()
    obj.fail = 1
    self.assertEqual(self.exceptions, [(obj, 'fail', 0, 1)])