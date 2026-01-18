import unittest
from zope.interface.tests import OptimizationTestMixin
def test__normalize_name_other(self):
    from zope.interface.adapter import _normalize_name
    for other in (1, 1.0, (), [], {}, object()):
        self.assertRaises(TypeError, _normalize_name, other)