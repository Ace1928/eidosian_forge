import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
def test__repr__implementedBy_in_interfaces(self):
    from zope.interface import Interface
    from zope.interface import implementedBy

    class IFoo(Interface):
        """Does nothing"""

    class Bar:
        """Does nothing"""
    impl = implementedBy(type(self))
    inst = self._makeOne(Bar, IFoo, impl)
    self.assertEqual(repr(inst), 'directlyProvides(Bar, IFoo, classImplements(TestProvidesClassRepr))')