import sys
import unittest.mock
import warnings
import weakref
from traits.api import HasTraits
from traits.constants import (
from traits.ctrait import CTrait
from traits.trait_errors import TraitError
from traits.trait_types import Any, Int, List
def test_initialization_with_keywords_fails(self):
    with self.assertRaises(TraitError):
        CTrait(kind=0)