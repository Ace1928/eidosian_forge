import pickle
import unittest
from unittest import mock
from traits.api import HasTraits, Set, Str
from traits.trait_base import _validate_everything
from traits.trait_errors import TraitError
from traits.trait_set_object import TraitSet, TraitSetEvent
from traits.trait_types import _validate_int
def test_add_unhashable(self):
    with self.assertRaises(TypeError) as python_e:
        set().add([])
    with self.assertRaises(TypeError) as trait_e:
        TraitSet().add([])
    self.assertEqual(str(trait_e.exception), str(python_e.exception))