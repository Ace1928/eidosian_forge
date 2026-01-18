import io
import linecache
import os
import sys
import sysconfig
import token
import tokenize
import inspect
import gc
import dis
import pickle
from time import monotonic as _time
import threading
def is_ignored_filename(self, filename):
    """Return True if the filename does not refer to a file
        we want to have reported.
        """
    return filename.startswith('<') and filename.endswith('>')