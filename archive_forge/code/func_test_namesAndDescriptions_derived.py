import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test_namesAndDescriptions_derived(self):
    from zope.interface import Attribute
    from zope.interface import Interface
    from zope.interface.interface import Method

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
    name_values = sorted(IDerived.namesAndDescriptions())
    self.assertEqual(len(name_values), 3)
    self.assertEqual(name_values[0][0], 'attr2')
    self.assertTrue(isinstance(name_values[0][1], Attribute))
    self.assertEqual(name_values[0][1].__name__, 'attr2')
    self.assertEqual(name_values[0][1].__doc__, 'My attr2')
    self.assertEqual(name_values[1][0], 'method')
    self.assertTrue(isinstance(name_values[1][1], Method))
    self.assertEqual(name_values[1][1].__name__, 'method')
    self.assertEqual(name_values[1][1].__doc__, 'My method, overridden')
    self.assertEqual(name_values[2][0], 'method2')
    self.assertTrue(isinstance(name_values[2][1], Method))
    self.assertEqual(name_values[2][1].__name__, 'method2')
    self.assertEqual(name_values[2][1].__doc__, 'My method2')
    name_values = sorted(IDerived.namesAndDescriptions(all=True))
    self.assertEqual(len(name_values), 4)
    self.assertEqual(name_values[0][0], 'attr')
    self.assertTrue(isinstance(name_values[0][1], Attribute))
    self.assertEqual(name_values[0][1].__name__, 'attr')
    self.assertEqual(name_values[0][1].__doc__, 'My attr')
    self.assertEqual(name_values[1][0], 'attr2')
    self.assertTrue(isinstance(name_values[1][1], Attribute))
    self.assertEqual(name_values[1][1].__name__, 'attr2')
    self.assertEqual(name_values[1][1].__doc__, 'My attr2')
    self.assertEqual(name_values[2][0], 'method')
    self.assertTrue(isinstance(name_values[2][1], Method))
    self.assertEqual(name_values[2][1].__name__, 'method')
    self.assertEqual(name_values[2][1].__doc__, 'My method, overridden')
    self.assertEqual(name_values[3][0], 'method2')
    self.assertTrue(isinstance(name_values[3][1], Method))
    self.assertEqual(name_values[3][1].__name__, 'method2')
    self.assertEqual(name_values[3][1].__doc__, 'My method2')