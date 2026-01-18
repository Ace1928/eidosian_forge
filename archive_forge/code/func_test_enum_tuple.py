import enum
import unittest
from traits.api import (
from traits.etsconfig.api import ETSConfig
from traits.testing.optional_dependencies import requires_traitsui
def test_enum_tuple(self):
    example = EnumTupleExample()
    self.assertEqual(example.value, 'foo')
    self.assertEqual(example.value_default, 'bar')
    self.assertEqual(example.value_name, 'foo')
    self.assertEqual(example.value_name_default, 'bar')
    example.value = 'bar'
    self.assertEqual(example.value, 'bar')
    with self.assertRaises(TraitError):
        example.value = 'something'
    with self.assertRaises(TraitError):
        example.value = 0
    example.values = ('one', 'two', 'three')
    example.value_name = 'two'
    self.assertEqual(example.value_name, 'two')
    with self.assertRaises(TraitError):
        example.value_name = 'bar'