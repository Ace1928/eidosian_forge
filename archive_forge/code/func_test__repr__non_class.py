import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
def test__repr__non_class(self):

    class Object:
        __bases__ = ()
        __str__ = lambda _: self.fail('Should not call str')

        def __repr__(self):
            return '<Object>'
    inst = self._makeOne(Object())
    self.assertEqual(repr(inst), 'directlyProvides(<Object>)')