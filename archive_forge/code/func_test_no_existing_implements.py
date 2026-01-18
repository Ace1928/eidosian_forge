import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
def test_no_existing_implements(self):
    from zope.interface.declarations import Implements
    from zope.interface.declarations import classImplements
    from zope.interface.interface import InterfaceClass
    IFoo = InterfaceClass('IFoo')

    class Foo:
        __implements_advice_data__ = ((IFoo,), classImplements)
    self._callFUT(Foo)
    self.assertNotIn('__implements_advice_data__', Foo.__dict__)
    self.assertIsInstance(Foo.__implemented__, Implements)
    self.assertEqual(list(Foo.__implemented__), [IFoo])