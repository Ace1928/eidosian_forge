import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test_interface_object_provides_Interface(self):
    from zope.interface import Interface

    class AnInterface(Interface):
        pass
    self.assertTrue(Interface.providedBy(AnInterface))