import gc
import unittest
from traits import trait_notifiers
from traits.api import Event, Float, HasTraits, List, on_trait_change
@on_trait_change('ok')
def method_listener_0(self):
    self.rebind_calls_0.append(True)