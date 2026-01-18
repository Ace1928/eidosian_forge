import os
import traceback
from _pydevd_bundle.pydevd_constants import ForkSafeLock
def debug_exception(msg=None):
    if DEBUG:
        if msg:
            debug(msg)
        with _debug_lock:
            with open(DEBUG_FILE, 'a+') as stream:
                _pid_prefix = _pid_msg
                if isinstance(msg, bytes):
                    _pid_prefix = _pid_prefix.encode('utf-8')
                stream.write(_pid_prefix)
                traceback.print_exc(file=stream)