import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
def test_dictless_w_existing_not_Implements(self):
    from zope.interface.interface import InterfaceClass

    class Foo:
        __slots__ = ('__implemented__',)
    foo = Foo()
    IFoo = InterfaceClass('IFoo')
    foo.__implemented__ = (IFoo,)
    self.assertEqual(list(self._callFUT(foo)), [IFoo])