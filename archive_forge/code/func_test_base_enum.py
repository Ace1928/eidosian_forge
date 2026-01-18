import enum
import unittest
from traits.api import (
from traits.etsconfig.api import ETSConfig
from traits.testing.optional_dependencies import requires_traitsui
def test_base_enum(self):
    obj = EnumCollectionExample()
    self.assertEqual(obj.slow_enum, 'yes')
    obj.slow_enum = 'no'
    self.assertEqual(obj.slow_enum, 'no')
    with self.assertRaises(TraitError):
        obj.slow_enum = 'perhaps'
    self.assertEqual(obj.slow_enum, 'no')