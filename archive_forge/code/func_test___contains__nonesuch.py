import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test___contains__nonesuch(self):
    from zope.interface import Interface

    class IEmpty(Interface):
        pass
    self.assertFalse('nonesuch' in IEmpty)