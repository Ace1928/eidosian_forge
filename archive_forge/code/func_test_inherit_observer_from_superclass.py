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
def test_inherit_observer_from_superclass(self):

    class BaseClass(HasTraits):
        events = List()

        @observe('value')
        def handler(self, event):
            self.events.append(event)

    class SubClass(BaseClass):
        value = Int()
    instance = SubClass()
    instance.value += 1
    self.assertEqual(len(instance.events), 1)