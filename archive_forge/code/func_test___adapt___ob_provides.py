import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test___adapt___ob_provides(self):
    ib = self._makeOne(True)
    adapted = object()
    self.assertIs(ib.__adapt__(adapted), adapted)