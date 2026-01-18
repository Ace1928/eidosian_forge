import gc
import sys
import unittest
from traits.constants import DefaultValue
from traits.has_traits import (
from traits.testing.optional_dependencies import numpy, requires_numpy
from traits.trait_errors import TraitError
from traits.trait_type import TraitType
from traits.trait_types import (
def test_no_leaking_notifiers(self):
    """ Extended trait change notifications should not leaf
        TraitChangeNotifyWrappers.
        """
    dummy = Dummy()
    ctrait = dummy._trait('x', 2)
    self.assertEqual(len(ctrait._notifiers(True)), 0)
    presenter = Presenter(obj=dummy)
    self.assertEqual(len(ctrait._notifiers(True)), 1)
    del presenter
    self.assertEqual(len(ctrait._notifiers(True)), 0)