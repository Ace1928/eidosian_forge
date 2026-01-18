from __future__ import print_function   # This version of olefile requires Python 2.7 or 3.5+.
import io
import sys
import struct, array, os.path, datetime, logging, warnings, traceback
def filetime2datetime(filetime):
    """
    convert FILETIME (64 bits int) to Python datetime.datetime
    """
    _FILETIME_null_date = datetime.datetime(1601, 1, 1, 0, 0, 0)
    return _FILETIME_null_date + datetime.timedelta(microseconds=filetime // 10)