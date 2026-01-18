import threading
import time
import unittest
from traits import trait_notifiers
from traits.api import Callable, Float, HasTraits, on_trait_change
def flush_event_loop(self):
    """ Post and process the Qt events. """
    qt4_app.sendPostedEvents()
    qt4_app.processEvents()