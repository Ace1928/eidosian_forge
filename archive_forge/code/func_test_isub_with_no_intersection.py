import pickle
import unittest
from unittest import mock
from traits.api import HasTraits, Set, Str
from traits.trait_base import _validate_everything
from traits.trait_errors import TraitError
from traits.trait_set_object import TraitSet, TraitSetEvent
from traits.trait_types import _validate_int
def test_isub_with_no_intersection(self):
    python_set = set([3, 4, 5])
    python_set -= set((i for i in range(2)))
    notifier = mock.Mock()
    ts = TraitSet((3, 4, 5), notifiers=[notifier])
    ts -= set((i for i in range(2)))
    self.assertEqual(ts, python_set)
    notifier.assert_not_called()