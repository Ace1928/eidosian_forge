import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
def test_dictless_w_existing_Implements(self):
    from zope.interface.declarations import Implements
    impl = Implements()

    class Foo:
        __slots__ = ('__implemented__',)
    foo = Foo()
    foo.__implemented__ = impl
    self.assertTrue(self._callFUT(foo) is impl)