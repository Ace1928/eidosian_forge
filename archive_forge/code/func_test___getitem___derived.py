import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test___getitem___derived(self):
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
    a_desc = IDerived['attr']
    self.assertTrue(isinstance(a_desc, Attribute))
    self.assertEqual(a_desc.__name__, 'attr')
    self.assertEqual(a_desc.__doc__, 'My attr')
    m_desc = IDerived['method']
    self.assertTrue(isinstance(m_desc, Method))
    self.assertEqual(m_desc.__name__, 'method')
    self.assertEqual(m_desc.__doc__, 'My method, overridden')
    a2_desc = IDerived['attr2']
    self.assertTrue(isinstance(a2_desc, Attribute))
    self.assertEqual(a2_desc.__name__, 'attr2')
    self.assertEqual(a2_desc.__doc__, 'My attr2')
    m2_desc = IDerived['method2']
    self.assertTrue(isinstance(m2_desc, Method))
    self.assertEqual(m2_desc.__name__, 'method2')
    self.assertEqual(m2_desc.__doc__, 'My method2')