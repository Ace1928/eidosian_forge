import gc
import unittest
from traits import trait_notifiers
from traits.api import Event, Float, HasTraits, List, on_trait_change
@on_trait_change('fail')
def failing_method_listener_3(self, obj, name, new):
    self.exceptions_from.append(3)
    raise Exception('error')