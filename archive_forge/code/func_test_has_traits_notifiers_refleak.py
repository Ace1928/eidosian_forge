import gc
import sys
import unittest
from traits.constants import DefaultValue
from traits.has_traits import (
from traits.testing.optional_dependencies import numpy, requires_numpy
from traits.trait_errors import TraitError
from traits.trait_type import TraitType
from traits.trait_types import (
def test_has_traits_notifiers_refleak(self):
    warmup = 5
    cycles = 10
    counts = []

    def handler():
        pass

    def f():
        obj = HasTraits()
        obj.on_trait_change(handler)
    for _ in range(cycles):
        f()
        gc.collect()
        counts.append(len(gc.get_objects()))
    self.assertEqual(counts[warmup:-1], counts[warmup + 1:])