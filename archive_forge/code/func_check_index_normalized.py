import copy
import operator
import pickle
import unittest.mock
from traits.api import HasTraits, Int, List
from traits.testing.optional_dependencies import numpy, requires_numpy
from traits.trait_base import _validate_everything
from traits.trait_errors import TraitError
from traits.trait_list_object import (
def check_index_normalized(self, index, length):
    if isinstance(index, slice):
        start, stop, step = (index.start, index.stop, index.step)
        self.assertIsNotNone(start)
        self.assertIsNotNone(stop)
        self.assertIsNotNone(step)
        self.assertTrue(0 <= start < stop <= length, msg='start and stop of {} not normalized for length {}'.format(index, length))
        self.assertTrue(step > 1, msg='step should be greater than 1')
        self.assertTrue(start + step < stop, msg='slice represents fewer than 2 elements')
        self.assertTrue((stop - start) % step == 1, msg='stop not normalised with respect to step')
    else:
        self.assertTrue(0 <= index <= length, msg='index {} is not normalized for length {}'.format(index, length))