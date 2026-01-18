import pickle
import unittest
from unittest import mock
from traits.api import HasTraits, Set, Str
from traits.trait_base import _validate_everything
from traits.trait_errors import TraitError
from traits.trait_set_object import TraitSet, TraitSetEvent
from traits.trait_types import _validate_int
def test_intersection_update_with_no_arguments(self):
    python_set = set([1, 2, 3])
    python_set.intersection_update()
    notifier = mock.Mock()
    ts = TraitSet([1, 2, 3], notifiers=[notifier])
    ts.intersection_update()
    self.assertEqual(ts, python_set)
    notifier.assert_not_called