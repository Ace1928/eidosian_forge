import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test_ctor_no_doc_space_in_name(self):
    element = self._makeOne('An Element')
    self.assertEqual(element.__name__, None)
    self.assertEqual(element.__doc__, 'An Element')