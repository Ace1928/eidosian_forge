import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
def test_builtins_w_existing_cache(self):
    from zope.interface import declarations
    t_spec, l_spec, d_spec = (object(), object(), object())
    with _MonkeyDict(declarations, 'BuiltinImplementationSpecifications') as specs:
        specs[tuple] = t_spec
        specs[list] = l_spec
        specs[dict] = d_spec
        self.assertTrue(self._callFUT(tuple) is t_spec)
        self.assertTrue(self._callFUT(list) is l_spec)
        self.assertTrue(self._callFUT(dict) is d_spec)