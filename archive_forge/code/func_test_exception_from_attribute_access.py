import sys
import unittest.mock
import warnings
import weakref
from traits.api import HasTraits
from traits.constants import (
from traits.ctrait import CTrait
from traits.trait_errors import TraitError
from traits.trait_types import Any, Int, List
def test_exception_from_attribute_access(self):
    self.assertFalse(hasattr(CTrait, 'badattr_test'))
    CTrait.badattr_test = property(lambda self: 1 / 0)
    try:
        ctrait = CTrait(0)
        with self.assertRaises(ZeroDivisionError):
            ctrait.badattr_test
    finally:
        del CTrait.badattr_test