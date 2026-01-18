import gc
import sys
import unittest
from traits.constants import DefaultValue
from traits.has_traits import (
from traits.testing.optional_dependencies import numpy, requires_numpy
from traits.trait_errors import TraitError
from traits.trait_type import TraitType
from traits.trait_types import (
def test_modify_nested_set_no_events(self):
    instance = NestedContainerClass(list_of_set=[set()])
    instance.on_trait_change(self.change_handler, 'list_of_set_items')
    instance.list_of_set[0].add(1)
    self.assertEqual(len(self.events), 0, 'Expected no events.')