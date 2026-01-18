import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test_ctor_attrs_w___locals__(self):
    ATTRS = {'__locals__': {}}
    klass = self._getTargetClass()
    inst = klass('ITesting', attrs=ATTRS)
    self.assertEqual(inst.__name__, 'ITesting')
    self.assertEqual(inst.__doc__, '')
    self.assertEqual(inst.__bases__, ())
    self.assertEqual(list(inst.names()), [])