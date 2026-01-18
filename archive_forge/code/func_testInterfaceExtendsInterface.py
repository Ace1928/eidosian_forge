import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def testInterfaceExtendsInterface(self):
    from zope.interface import Interface
    new = Interface.__class__
    FunInterface = new('FunInterface')
    BarInterface = new('BarInterface', (FunInterface,))
    BobInterface = new('BobInterface')
    BazInterface = new('BazInterface', (BobInterface, BarInterface))
    self.assertTrue(BazInterface.extends(BobInterface))
    self.assertTrue(BazInterface.extends(BarInterface))
    self.assertTrue(BazInterface.extends(FunInterface))
    self.assertFalse(BobInterface.extends(FunInterface))
    self.assertFalse(BobInterface.extends(BarInterface))
    self.assertTrue(BarInterface.extends(FunInterface))
    self.assertFalse(BarInterface.extends(BazInterface))