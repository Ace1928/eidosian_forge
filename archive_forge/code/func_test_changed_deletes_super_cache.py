import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
def test_changed_deletes_super_cache(self):
    impl = self._makeOne()
    self.assertIsNone(impl._super_cache)
    self.assertNotIn('_super_cache', impl.__dict__)
    impl._super_cache = 42
    self.assertIn('_super_cache', impl.__dict__)
    impl.changed(None)
    self.assertIsNone(impl._super_cache)
    self.assertNotIn('_super_cache', impl.__dict__)