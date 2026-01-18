import unittest
from traits import trait_notifiers
from traits.api import Float, HasTraits, Undefined
def test_anytrait_static_notifiers_1(self):

    class AnytraitStaticNotifiers1(HasTraits):
        ok = Float

        def _anytrait_changed(self):
            if not hasattr(self, 'anycalls'):
                self.anycalls = []
            self.anycalls.append(True)
    obj = AnytraitStaticNotifiers1(ok=2)
    obj.ok = 3
    self.assertEqual(len(obj.anycalls), 3)