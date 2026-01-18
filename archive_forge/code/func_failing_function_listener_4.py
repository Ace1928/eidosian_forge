import gc
import unittest
from traits import trait_notifiers
from traits.api import Event, Float, HasTraits, List, on_trait_change
def failing_function_listener_4(obj, name, old, new):
    exceptions_from.append(4)
    raise Exception('error')