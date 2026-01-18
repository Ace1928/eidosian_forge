import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test___iter__simple(self):
    from zope.interface import Attribute
    from zope.interface import Interface

    class ISimple(Interface):
        attr = Attribute('My attr')

        def method():
            """My method"""
    self.assertEqual(sorted(list(ISimple)), ['attr', 'method'])