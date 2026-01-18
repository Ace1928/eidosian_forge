import copy
import operator
import pickle
import unittest.mock
from traits.api import HasTraits, Int, List
from traits.testing.optional_dependencies import numpy, requires_numpy
from traits.trait_base import _validate_everything
from traits.trait_errors import TraitError
from traits.trait_list_object import (
def test_init_from_iterable(self):

    class Foo:
        pass
    tl = TraitListObject(trait=List(), object=Foo(), name='foo', value=squares(5))
    self.assertEqual(tl, list(squares(5)))