import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test_classImplements_base_not_derived(self):
    from zope.interface import Interface
    from zope.interface import implementedBy
    from zope.interface import providedBy

    class IBase(Interface):

        def method():
            """docstring"""

    class IDerived(IBase):
        pass

    class Current:
        __implemented__ = IBase

        def method(self):
            raise NotImplementedError()
    current = Current()
    self.assertTrue(IBase.implementedBy(Current))
    self.assertFalse(IDerived.implementedBy(Current))
    self.assertTrue(IBase in implementedBy(Current))
    self.assertFalse(IDerived in implementedBy(Current))
    self.assertTrue(IBase in providedBy(current))
    self.assertFalse(IDerived in providedBy(current))