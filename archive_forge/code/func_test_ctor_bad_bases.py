import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test_ctor_bad_bases(self):
    klass = self._getTargetClass()
    self.assertRaises(TypeError, klass, 'ITesting', (object(),))