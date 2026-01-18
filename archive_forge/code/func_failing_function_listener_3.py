import gc
import unittest
from traits import trait_notifiers
from traits.api import Event, Float, HasTraits, List, on_trait_change
def failing_function_listener_3(obj, name, new):
    exceptions_from.append(3)
    raise Exception('error')