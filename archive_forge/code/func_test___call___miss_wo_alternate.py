import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test___call___miss_wo_alternate(self):
    from zope.interface import Interface

    class I(Interface):
        pass

    class C:
        pass
    c = C()
    self.assertRaises(TypeError, I, c)