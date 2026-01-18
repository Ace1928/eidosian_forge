import gc
import unittest
from traits import trait_notifiers
from traits.api import Event, Float, HasTraits, List, on_trait_change
def function_listener_4(obj, name, old, new):
    calls_4.append((obj, name, old, new))