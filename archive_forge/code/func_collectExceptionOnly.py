import sys
import traceback
import time
from io import StringIO
import linecache
from paste.exceptions import serial_number_generator
import warnings
def collectExceptionOnly(self, etype, value):
    return traceback.format_exception_only(etype, value)