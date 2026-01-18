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
def test_deepcopy_memoization(self):

    class A(HasTraits):
        x = Int()
        y = Str()
    a = A()
    objs = [a, a]
    objs_copy = copy.deepcopy(objs)
    self.assertIsNot(objs_copy[0], objs[0])
    self.assertIs(objs_copy[0], objs_copy[1])