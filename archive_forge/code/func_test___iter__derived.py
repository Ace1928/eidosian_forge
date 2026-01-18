import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test___iter__derived(self):
    from zope.interface import Attribute
    from zope.interface import Interface

    class IBase(Interface):
        attr = Attribute('My attr')

        def method():
            """My method"""

    class IDerived(IBase):
        attr2 = Attribute('My attr2')

        def method():
            """My method, overridden"""

        def method2():
            """My method2"""
    self.assertEqual(sorted(list(IDerived)), ['attr', 'attr2', 'method', 'method2'])