import os
import sys
import traceback
from _pydev_bundle.pydev_imports import xmlrpclib, _queue, Exec
from  _pydev_bundle._pydev_calltip_util import get_description
from _pydevd_bundle import pydevd_vars
from _pydevd_bundle import pydevd_xml
from _pydevd_bundle.pydevd_constants import (IS_JYTHON, NEXT_VALUE_SEPARATOR, get_global_debugger,
from contextlib import contextmanager
from _pydev_bundle import pydev_log
from _pydevd_bundle.pydevd_utils import interrupt_main_thread
from io import StringIO
def do_enable_gui():
    from _pydev_bundle.pydev_versioncheck import versionok_for_gui
    if versionok_for_gui():
        try:
            from pydev_ipython.inputhook import enable_gui
            enable_gui(guiname)
        except:
            sys.stderr.write("Failed to enable GUI event loop integration for '%s'\n" % guiname)
            traceback.print_exc()
    elif guiname not in ['none', '', None]:
        sys.stderr.write("PyDev console: Python version does not support GUI event loop integration for '%s'\n" % guiname)
    return guiname