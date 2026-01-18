import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test___getitem___miss(self):
    one = self._makeOne()

    def _test():
        return one['nonesuch']
    self.assertRaises(KeyError, _test)