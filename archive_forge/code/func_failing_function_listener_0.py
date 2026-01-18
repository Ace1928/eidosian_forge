import gc
import unittest
from traits import trait_notifiers
from traits.api import Event, Float, HasTraits, List, on_trait_change
def failing_function_listener_0():
    exceptions_from.append(0)
    raise Exception('error')