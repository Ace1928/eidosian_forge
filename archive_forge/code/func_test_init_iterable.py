import copy
import pickle
import sys
import unittest
from unittest import mock
from traits.api import HasTraits
from traits.trait_dict_object import TraitDict, TraitDictEvent, TraitDictObject
from traits.trait_errors import TraitError
from traits.trait_types import Dict, Int, Str
def test_init_iterable(self):
    td = TraitDict([('a', 1), ('b', 2)], key_validator=str_validator, value_validator=int_validator)
    self.assertEqual(td, {'a': 1, 'b': 2})
    self.assertEqual(td.notifiers, [])
    with self.assertRaises(ValueError):
        TraitDict(['a', 'b'], key_validator=str_validator, value_validator=int_validator)