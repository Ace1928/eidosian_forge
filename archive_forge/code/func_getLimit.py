import sys
import traceback
import time
from io import StringIO
import linecache
from paste.exceptions import serial_number_generator
import warnings
def getLimit(self):
    limit = self.limit
    if limit is None:
        limit = getattr(sys, 'tracebacklimit', None)
    return limit