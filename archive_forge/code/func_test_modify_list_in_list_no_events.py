import gc
import sys
import unittest
from traits.constants import DefaultValue
from traits.has_traits import (
from traits.testing.optional_dependencies import numpy, requires_numpy
from traits.trait_errors import TraitError
from traits.trait_type import TraitType
from traits.trait_types import (
def test_modify_list_in_list_no_events(self):
    instance = NestedContainerClass(list_of_list=[[]])
    instance.on_trait_change(self.change_handler, 'list_of_list_items')
    instance.list_of_list[0].append(1)
    self.assertEqual(len(self.events), 0, 'Expected no events.')