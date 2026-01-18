import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test_implementedBy_miss(self):
    from zope.interface import interface
    from zope.interface.declarations import _empty
    sb = self._makeOne()

    def _implementedBy(obj):
        return _empty
    with _Monkey(interface, implementedBy=_implementedBy):
        self.assertFalse(sb.implementedBy(object()))