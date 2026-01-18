import copy
import operator
import pickle
import unittest.mock
from traits.api import HasTraits, Int, List
from traits.testing.optional_dependencies import numpy, requires_numpy
from traits.trait_base import _validate_everything
from traits.trait_errors import TraitError
from traits.trait_list_object import (
def test_list_of_lists_pickle_with_notifier(self):

    class Foo:
        pass
    tl = TraitListObject(trait=List(), object=Foo(), name='foo', value=())
    self.assertEqual([tl.notifier], tl.notifiers)
    serialized = pickle.dumps(tl)
    tl_deserialized = pickle.loads(serialized)
    self.assertEqual([tl_deserialized.notifier], tl_deserialized.notifiers)