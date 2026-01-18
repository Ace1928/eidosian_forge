import copy
import pickle
import unittest
from traits.has_traits import (
from traits.ctrait import CTrait
from traits.observation.api import (
from traits.observation.exception_handling import (
from traits.traits import ForwardProperty, generic_trait
from traits.trait_types import Event, Float, Instance, Int, List, Map, Str
from traits.trait_errors import TraitError
def test__trait_set_inited(self):
    foo = HasTraits.__new__(HasTraits)
    self.assertFalse(foo.traits_inited())
    foo._trait_set_inited()
    self.assertTrue(foo.traits_inited())