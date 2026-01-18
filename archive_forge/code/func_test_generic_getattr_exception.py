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
def test_generic_getattr_exception(self):

    class PropertyLike:
        """
            Data descriptor giving a property-like object that produces
            successive reciprocals on __get__. This means that it raises
            on first access, but not on subsequent accesses.
            """

        def __init__(self):
            self.n = 0

        def __get__(self, obj, type=None):
            old_n = self.n
            self.n += 1
            return 1 / old_n

        def __set__(self, obj, value):
            raise AttributeError('Read-only descriptor')

    class A(HasTraits):
        fruit = PropertyLike()
        banana_ = Int(1729)
    a = A()
    with self.assertRaises(ZeroDivisionError):
        a.fruit
    with self.assertRaises(AttributeError):
        a.veg
    self.assertEqual(a.banananana, 1729)