import gc
import unittest
from traits import trait_notifiers
from traits.api import Event, Float, HasTraits, List, on_trait_change
def failing_function_listener_2(name, new):
    exceptions_from.append(2)
    raise Exception('error')