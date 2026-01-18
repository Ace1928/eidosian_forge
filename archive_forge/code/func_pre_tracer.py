import inspect
import os
import threading
from contextlib import contextmanager
from datetime import datetime
from traits import trait_notifiers
def pre_tracer(self, obj, name, old, new, handler):
    """ The traits pre event tracer.

        This method should be set as the global pre event tracer for traits.

        """
    tracer = self._get_tracer()
    tracer.pre_tracer(obj, name, old, new, handler)