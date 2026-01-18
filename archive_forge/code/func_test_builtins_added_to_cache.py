import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
def test_builtins_added_to_cache(self):
    from zope.interface import declarations
    from zope.interface.declarations import Implements
    with _MonkeyDict(declarations, 'BuiltinImplementationSpecifications') as specs:
        self.assertEqual(list(self._callFUT(tuple)), [])
        self.assertEqual(list(self._callFUT(list)), [])
        self.assertEqual(list(self._callFUT(dict)), [])
        for typ in (tuple, list, dict):
            spec = specs[typ]
            self.assertIsInstance(spec, Implements)
            self.assertEqual(repr(spec), 'classImplements(%s)' % (typ.__name__,))