import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test___doc___as_element(self):
    from zope.interface import Attribute
    from zope.interface import Interface

    class IHaveADocString(Interface):
        """xxx"""
        __doc__ = Attribute('the doc')
    self.assertEqual(IHaveADocString.__doc__, '')
    self.assertEqual(list(IHaveADocString), ['__doc__'])