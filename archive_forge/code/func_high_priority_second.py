import gc
import unittest
from traits import trait_notifiers
from traits.api import Event, Float, HasTraits, List, on_trait_change
def high_priority_second(self):
    self.prioritized_notifications.append(3)