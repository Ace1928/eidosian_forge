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
def do_exec_code(self, code, is_single_line):
    try:
        code_fragment = CodeFragment(code, is_single_line)
        more = self.need_more(code_fragment)
        if not more:
            code_fragment = self.buffer
            self.buffer = None
            self.exec_queue.put(code_fragment)
        return more
    except:
        traceback.print_exc()
        return False