import os
import sys
import threading
from . import process
from . import reduction
def freeze_support(self):
    """Check whether this is a fake forked process in a frozen executable.
        If so then run code specified by commandline and exit.
        """
    if sys.platform == 'win32' and getattr(sys, 'frozen', False):
        from .spawn import freeze_support
        freeze_support()