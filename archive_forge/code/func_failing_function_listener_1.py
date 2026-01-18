import gc
import unittest
from traits import trait_notifiers
from traits.api import Event, Float, HasTraits, List, on_trait_change
def failing_function_listener_1(new):
    exceptions_from.append(1)
    raise Exception('error')