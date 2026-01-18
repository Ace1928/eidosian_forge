import copy
import operator
import pickle
import unittest.mock
from traits.api import HasTraits, Int, List
from traits.testing.optional_dependencies import numpy, requires_numpy
from traits.trait_base import _validate_everything
from traits.trait_errors import TraitError
from traits.trait_list_object import (
def test_length_violation_error_message(self):
    foo = HasLengthConstrainedLists(at_least_two=[1, 2])
    with self.assertRaises(TraitError) as exc_cm:
        foo.at_least_two.remove(1)
    exc_message = str(exc_cm.exception)
    self.assertIn("'at_least_two' trait", exc_message)
    self.assertIn('HasLengthConstrainedLists instance', exc_message)
    self.assertIn('an integer', exc_message)
    self.assertIn('at least 2 items', exc_message)