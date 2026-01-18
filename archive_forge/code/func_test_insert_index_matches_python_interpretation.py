import copy
import operator
import pickle
import unittest.mock
from traits.api import HasTraits, Int, List
from traits.testing.optional_dependencies import numpy, requires_numpy
from traits.trait_base import _validate_everything
from traits.trait_errors import TraitError
from traits.trait_list_object import (
def test_insert_index_matches_python_interpretation(self):
    for insertion_index in range(-10, 10):
        with self.subTest(insertion_index=insertion_index):
            tl = TraitList([5, 6, 7])
            pl = [5, 6, 7]
            tl.insert(insertion_index, 1729)
            pl.insert(insertion_index, 1729)
            self.assertEqual(tl, pl)