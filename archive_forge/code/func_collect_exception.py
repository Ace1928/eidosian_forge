import sys
import traceback
import time
from io import StringIO
import linecache
from paste.exceptions import serial_number_generator
import warnings
def collect_exception(t, v, tb, limit=None):
    """
    Collection an exception from ``sys.exc_info()``.

    Use like::

      try:
          blah blah
      except:
          exc_data = collect_exception(*sys.exc_info())
    """
    return col.collectException(t, v, tb, limit=limit)