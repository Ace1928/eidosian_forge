import gc
import unittest
from traits import trait_notifiers
from traits.api import Event, Float, HasTraits, List, on_trait_change
def test_creating_notifiers_dont_create_cyclic_garbage(self):
    gc.collect()
    DynamicNotifiers()
    self.assertEqual(gc.collect(), 0)