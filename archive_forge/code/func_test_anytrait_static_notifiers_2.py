import unittest
from traits import trait_notifiers
from traits.api import Float, HasTraits, Undefined
def test_anytrait_static_notifiers_2(self):

    class AnytraitStaticNotifiers2(HasTraits):
        ok = Float

        def _anytrait_changed(self, name):
            if not hasattr(self, 'anycalls'):
                self.anycalls = []
            self.anycalls.append(name)
    obj = AnytraitStaticNotifiers2(ok=2)
    obj.ok = 3
    expected = ['trait_added', 'ok', 'ok']
    self.assertEqual(expected, obj.anycalls)