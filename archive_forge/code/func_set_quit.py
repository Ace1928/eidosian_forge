import fnmatch
import sys
import os
from inspect import CO_GENERATOR, CO_COROUTINE, CO_ASYNC_GENERATOR
def set_quit(self):
    """Set quitting attribute to True.

        Raises BdbQuit exception in the next call to a dispatch_*() method.
        """
    self.stopframe = self.botframe
    self.returnframe = None
    self.quitting = True
    sys.settrace(None)