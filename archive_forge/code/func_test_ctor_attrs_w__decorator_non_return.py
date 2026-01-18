import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test_ctor_attrs_w__decorator_non_return(self):
    from zope.interface.interface import _decorator_non_return
    ATTRS = {'dropme': _decorator_non_return}
    klass = self._getTargetClass()
    inst = klass('ITesting', attrs=ATTRS)
    self.assertEqual(inst.__name__, 'ITesting')
    self.assertEqual(inst.__doc__, '')
    self.assertEqual(inst.__bases__, ())
    self.assertEqual(list(inst.names()), [])