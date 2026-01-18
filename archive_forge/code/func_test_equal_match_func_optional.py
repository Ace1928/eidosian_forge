import unittest
from unittest import mock
from traits.has_traits import HasTraits
from traits.observation._testing import (
from traits.observation._trait_added_observer import (
from traits.trait_types import Str
def test_equal_match_func_optional(self):
    observer1 = TraitAddedObserver(match_func=DummyMatchFunc(return_value=True), optional=False)
    observer2 = TraitAddedObserver(match_func=DummyMatchFunc(return_value=True), optional=False)
    self.assertEqual(observer1, observer2)
    self.assertEqual(hash(observer1), hash(observer2))