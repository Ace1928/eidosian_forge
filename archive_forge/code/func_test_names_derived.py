import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test_names_derived(self):
    from zope.interface import Attribute
    from zope.interface import Interface

    class IBase(Interface):
        attr = Attribute('My attr')

        def method():
            """docstring"""

    class IDerived(IBase):
        attr2 = Attribute('My attr2')

        def method():
            """docstring"""

        def method2():
            """docstring"""
    self.assertEqual(sorted(IDerived.names()), ['attr2', 'method', 'method2'])
    self.assertEqual(sorted(IDerived.names(all=True)), ['attr', 'attr2', 'method', 'method2'])