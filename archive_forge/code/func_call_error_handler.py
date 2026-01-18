from array import array
import struct
import sys
import traceback
import types
from Xlib import X
from Xlib.support import lock
def call_error_handler(handler, error, request):
    try:
        return handler(error, request)
    except:
        sys.stderr.write('Exception raised by error handler.\n')
        traceback.print_exc()
        return 0