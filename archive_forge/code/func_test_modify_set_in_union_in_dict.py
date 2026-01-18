import gc
import sys
import unittest
from traits.constants import DefaultValue
from traits.has_traits import (
from traits.testing.optional_dependencies import numpy, requires_numpy
from traits.trait_errors import TraitError
from traits.trait_type import TraitType
from traits.trait_types import (
def test_modify_set_in_union_in_dict(self):
    instance = NestedContainerClass(dict_of_union_none_or_set={'1': set()})
    try:
        instance.dict_of_union_none_or_set['1'].add(1)
    except Exception:
        self.fail('Mutating a nested set should not fail.')