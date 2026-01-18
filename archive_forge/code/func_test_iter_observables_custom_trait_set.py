import unittest
from unittest import mock
from traits.has_traits import HasTraits
from traits.observation._set_item_observer import SetItemObserver
from traits.observation._testing import (
from traits.trait_set_object import TraitSet
from traits.trait_types import Set
def test_iter_observables_custom_trait_set(self):
    custom_trait_set = CustomTraitSet([1, 2, 3])
    observer = create_observer()
    actual = list(observer.iter_objects(custom_trait_set))
    self.assertCountEqual(actual, [1, 2, 3])