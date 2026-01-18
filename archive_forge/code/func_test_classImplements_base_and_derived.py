import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test_classImplements_base_and_derived(self):
    from zope.interface import Interface
    from zope.interface import implementedBy
    from zope.interface import providedBy

    class IBase(Interface):

        def method():
            """docstring"""

    class IDerived(IBase):
        pass

    class Current:
        __implemented__ = IDerived

        def method(self):
            raise NotImplementedError()
    current = Current()
    self.assertTrue(IBase.implementedBy(Current))
    self.assertTrue(IDerived.implementedBy(Current))
    self.assertFalse(IBase in implementedBy(Current))
    self.assertTrue(IBase in implementedBy(Current).flattened())
    self.assertTrue(IDerived in implementedBy(Current))
    self.assertFalse(IBase in providedBy(current))
    self.assertTrue(IBase in providedBy(current).flattened())
    self.assertTrue(IDerived in providedBy(current))