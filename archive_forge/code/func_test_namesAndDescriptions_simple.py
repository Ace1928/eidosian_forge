import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test_namesAndDescriptions_simple(self):
    from zope.interface import Attribute
    from zope.interface import Interface
    from zope.interface.interface import Method

    class ISimple(Interface):
        attr = Attribute('My attr')

        def method():
            """My method"""
    name_values = sorted(ISimple.namesAndDescriptions())
    self.assertEqual(len(name_values), 2)
    self.assertEqual(name_values[0][0], 'attr')
    self.assertTrue(isinstance(name_values[0][1], Attribute))
    self.assertEqual(name_values[0][1].__name__, 'attr')
    self.assertEqual(name_values[0][1].__doc__, 'My attr')
    self.assertEqual(name_values[1][0], 'method')
    self.assertTrue(isinstance(name_values[1][1], Method))
    self.assertEqual(name_values[1][1].__name__, 'method')
    self.assertEqual(name_values[1][1].__doc__, 'My method')