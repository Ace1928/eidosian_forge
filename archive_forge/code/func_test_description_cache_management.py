import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test_description_cache_management(self):
    from zope.interface import Attribute
    from zope.interface import Interface

    class I1(Interface):
        a = Attribute('a')

    class I2(I1):
        pass

    class I3(I2):
        pass
    self.assertTrue(I3.get('a') is I1.get('a'))
    I2.__bases__ = (Interface,)
    self.assertTrue(I3.get('a') is None)