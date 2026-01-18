import unittest
from zope.interface.tests import OptimizationTestMixin
def test_changed_w_subregistries(self):
    base = self._makeOne()

    class Derived:
        _changed = None

        def changed(self, originally_changed):
            self._changed = originally_changed
    derived1, derived2 = (Derived(), Derived())
    base._addSubregistry(derived1)
    base._addSubregistry(derived2)
    orig = object()
    base.changed(orig)
    self.assertIs(derived1._changed, orig)
    self.assertIs(derived2._changed, orig)