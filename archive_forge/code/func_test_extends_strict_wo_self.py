import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test_extends_strict_wo_self(self):
    from zope.interface.interface import Interface

    class IFoo(Interface):
        pass
    spec = self._makeOne(IFoo)
    self.assertFalse(spec.extends(IFoo, strict=True))