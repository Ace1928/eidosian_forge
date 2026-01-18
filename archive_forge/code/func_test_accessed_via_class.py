import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
def test_accessed_via_class(self):
    from zope.interface.declarations import Provides
    from zope.interface.interface import InterfaceClass
    IFoo = InterfaceClass('IFoo')

    class Foo:
        pass
    Foo.__provides__ = Provides(Foo, IFoo)
    Foo.__providedBy__ = self._makeOne()
    self.assertEqual(list(Foo.__providedBy__), [IFoo])