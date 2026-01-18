import pickle
import unittest
from unittest import mock
from traits.api import HasTraits, Set, Str
from traits.trait_base import _validate_everything
from traits.trait_errors import TraitError
from traits.trait_set_object import TraitSet, TraitSetEvent
from traits.trait_types import _validate_int
def test_ixor_no_nofications_for_no_change(self):
    notifier = mock.Mock()
    ts_1 = TraitSet([1, 2], notifiers=[notifier])
    ts_1 ^= set()
    notifier.assert_not_called()