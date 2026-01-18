import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test___iter__empty(self):
    from zope.interface import Interface

    class IEmpty(Interface):
        pass
    self.assertEqual(list(IEmpty), [])