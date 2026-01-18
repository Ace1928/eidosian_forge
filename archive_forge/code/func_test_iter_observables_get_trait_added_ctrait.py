import unittest
from unittest import mock
from traits.has_traits import HasTraits
from traits.observation._testing import (
from traits.observation._trait_added_observer import (
from traits.trait_types import Str
def test_iter_observables_get_trait_added_ctrait(self):
    observer = create_observer()
    instance = DummyHasTraitsClass()
    actual, = list(observer.iter_observables(instance))
    self.assertEqual(actual, instance._trait('trait_added', 2))