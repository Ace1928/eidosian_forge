import inspect
import os
import threading
from contextlib import contextmanager
from datetime import datetime
from traits import trait_notifiers
def post_tracer(self, obj, name, old, new, handler, exception=None):
    """ The traits post event tracer.

        This method should be set as the global post event tracer for traits.

        """
    tracer = self._get_tracer()
    tracer.post_tracer(obj, name, old, new, handler, exception=exception)