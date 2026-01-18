import gc
import unittest
from traits import trait_notifiers
from traits.api import Event, Float, HasTraits, List, on_trait_change
@on_trait_change('ok')
def method_listener_4(self, obj, name, old, new):
    self.rebind_calls_4.append((obj, name, old, new))