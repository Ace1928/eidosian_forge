from contextlib import contextmanager
import unittest
from traits.ctrait import CTrait
from traits.trait_converters import (
from traits.trait_factory import TraitFactory
from traits.trait_handlers import TraitCastType, TraitInstance
from traits.api import Any, Int
def test_check_trait_trait_type_class(self):
    result = check_trait(Int)
    self.assertIsInstance(result, CTrait)
    self.assertIsInstance(result.handler, Int)