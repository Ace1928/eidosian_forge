from collections import namedtuple
import dis
import os
import re
import sys
from _pydev_bundle._pydev_saved_modules import threading
from types import CodeType, FrameType
from typing import Dict, Optional, Tuple, Any
from os.path import basename, splitext
from _pydev_bundle import pydev_log
from _pydevd_bundle import pydevd_dont_trace
from _pydevd_bundle.pydevd_constants import (GlobalDebuggerHolder, ForkSafeLock,
from pydevd_file_utils import (NORM_PATHS_AND_BASE_CONTAINER,
from _pydevd_bundle.pydevd_trace_dispatch import should_stop_on_exception, handle_exception
from _pydevd_bundle.pydevd_constants import EXCEPTION_TYPE_HANDLED
from _pydevd_bundle.pydevd_trace_dispatch import is_unhandled_exception
from _pydevd_bundle.pydevd_breakpoints import stop_on_unhandled_exception
from _pydevd_bundle.pydevd_utils import get_clsname_for_code
from _pydevd_bundle.pydevd_additional_thread_info import set_additional_thread_info, any_thread_stepping, PyDBAdditionalThreadInfo
def stop_monitoring(all_threads=False):
    if all_threads:
        if monitor.get_tool(monitor.DEBUGGER_ID) == 'pydevd':
            monitor.set_events(monitor.DEBUGGER_ID, 0)
            monitor.register_callback(DEBUGGER_ID, monitor.events.PY_START, None)
            monitor.register_callback(DEBUGGER_ID, monitor.events.PY_RESUME, None)
            monitor.register_callback(DEBUGGER_ID, monitor.events.LINE, None)
            monitor.register_callback(DEBUGGER_ID, monitor.events.JUMP, None)
            monitor.register_callback(DEBUGGER_ID, monitor.events.PY_RETURN, None)
            monitor.register_callback(DEBUGGER_ID, monitor.events.RAISE, None)
            monitor.free_tool_id(monitor.DEBUGGER_ID)
    else:
        try:
            thread_info = _thread_local_info.thread_info
        except:
            thread_info = _get_thread_info(False, 1)
            if thread_info is None:
                return
        thread_info.trace = False