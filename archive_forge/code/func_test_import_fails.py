import unittest
from traits.testing.optional_dependencies import optional_import
def test_import_fails(self):
    module = optional_import('unavailable_module')
    self.assertIsNone(module)