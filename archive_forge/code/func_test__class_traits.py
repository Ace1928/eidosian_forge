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
def test__class_traits(self):

    class Base(HasTraits):
        pin = Int
    a = Base()
    a_class_traits = a._class_traits()
    self.assertIsInstance(a_class_traits, dict)
    self.assertIn('pin', a_class_traits)
    self.assertIsInstance(a_class_traits['pin'], CTrait)
    b = Base()
    self.assertIs(b._class_traits(), a_class_traits)