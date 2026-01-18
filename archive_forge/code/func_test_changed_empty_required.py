import unittest
from zope.interface.tests import OptimizationTestMixin
def test_changed_empty_required(self):

    class Mixin:

        def changed(self, *other):
            pass

    class Derived(self._getTargetClass(), Mixin):
        pass
    registry = self._makeRegistry()
    alb = Derived(registry)
    alb.changed(alb)