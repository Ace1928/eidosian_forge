from sys import version_info as _swig_python_version_info
import weakref
import inspect
import os
import re
import sys
import traceback
import inspect
import io
import os
import sys
import traceback
import types
def fz_defer_reap_start():
    """
    Class-aware wrapper for `::fz_defer_reap_start()`.
    	Increment the defer reap count.

    	No reap operations will take place (except for those
    	triggered by an immediate failed malloc) until the
    	defer reap count returns to 0.

    	Call this at the start of a process during which you
    	potentially might drop many reapable objects.

    	It is vital that every fz_defer_reap_start is matched
    	by a fz_defer_reap_end call.
    """
    return _mupdf.fz_defer_reap_start()