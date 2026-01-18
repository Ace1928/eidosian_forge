import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
def test_ctor_w_implements_in_bases(self):
    from zope.interface.declarations import Implements
    impl = Implements()
    decl = self._makeOne(impl)
    self.assertEqual(list(decl.__bases__), [impl])