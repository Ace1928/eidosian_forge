import sys
import bisect
import types
from _pydev_bundle._pydev_saved_modules import threading
from _pydevd_bundle import pydevd_utils, pydevd_source_mapping
from _pydevd_bundle.pydevd_additional_thread_info import set_additional_thread_info
from _pydevd_bundle.pydevd_comm import (InternalGetThreadStack, internal_get_completions,
from _pydevd_bundle.pydevd_comm_constants import (CMD_THREAD_SUSPEND, file_system_encoding,
from _pydevd_bundle.pydevd_constants import (get_current_thread_id, set_protocol, get_protocol,
from _pydevd_bundle.pydevd_net_command_factory_json import NetCommandFactoryJson
from _pydevd_bundle.pydevd_net_command_factory_xml import NetCommandFactory
import pydevd_file_utils
from _pydev_bundle import pydev_log
from _pydevd_bundle.pydevd_breakpoints import LineBreakpoint
from pydevd_tracing import get_exception_traceback_str
import os
import subprocess
import ctypes
from _pydevd_bundle.pydevd_collect_bytecode_info import code_to_bytecode_representation
import itertools
import linecache
from _pydevd_bundle.pydevd_utils import DAPGrouper, interrupt_main_thread
from _pydevd_bundle.pydevd_daemon_thread import run_as_pydevd_daemon_thread
from _pydevd_bundle.pydevd_thread_lifecycle import pydevd_find_thread_by_id, resume_threads
import tokenize
def remove_python_exception_breakpoint(self, py_db, exception):
    try:
        cp = py_db.break_on_uncaught_exceptions.copy()
        cp.pop(exception, None)
        py_db.break_on_uncaught_exceptions = cp
        cp = py_db.break_on_caught_exceptions.copy()
        cp.pop(exception, None)
        py_db.break_on_caught_exceptions = cp
        cp = py_db.break_on_user_uncaught_exceptions.copy()
        cp.pop(exception, None)
        py_db.break_on_user_uncaught_exceptions = cp
    except:
        pydev_log.exception('Error while removing exception %s', sys.exc_info()[0])
    py_db.on_breakpoints_changed(removed=True)