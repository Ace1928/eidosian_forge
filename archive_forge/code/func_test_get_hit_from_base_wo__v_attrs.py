import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test_get_hit_from_base_wo__v_attrs(self):
    from zope.interface.interface import Attribute
    from zope.interface.interface import Interface

    class IFoo(Interface):
        foo = Attribute('foo')

    class IBar(Interface):
        bar = Attribute('bar')
    spec = self._makeOne([IFoo, IBar])
    self.assertTrue(spec.get('foo') is IFoo.get('foo'))
    self.assertTrue(spec.get('bar') is IBar.get('bar'))