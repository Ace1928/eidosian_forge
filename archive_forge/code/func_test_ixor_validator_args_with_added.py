import pickle
import unittest
from unittest import mock
from traits.api import HasTraits, Set, Str
from traits.trait_base import _validate_everything
from traits.trait_errors import TraitError
from traits.trait_set_object import TraitSet, TraitSetEvent
from traits.trait_types import _validate_int
def test_ixor_validator_args_with_added(self):
    validator = mock.Mock(wraps=str)
    ts = TraitSet([1, 2, 3], item_validator=validator, notifiers=[self.notification_handler])
    self.assertEqual(ts, set(['1', '2', '3']))
    validator.reset_mock()
    ts ^= set(['2', 3, 4])
    validator_inputs = set((value for (value,), _ in validator.call_args_list))
    self.assertEqual(validator_inputs, set([3, 4]))
    self.assertEqual(ts, set(['1', '3', '4']))
    self.assertEqual(self.added, set(['4']))
    self.assertEqual(self.removed, set(['2']))