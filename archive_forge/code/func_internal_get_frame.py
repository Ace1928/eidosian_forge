import linecache
import os
from _pydev_bundle.pydev_imports import _queue
from _pydev_bundle._pydev_saved_modules import time
from _pydev_bundle._pydev_saved_modules import threading
from _pydev_bundle._pydev_saved_modules import socket as socket_module
from _pydevd_bundle.pydevd_constants import (DebugInfoHolder, IS_WINDOWS, IS_JYTHON, IS_WASM,
from _pydev_bundle.pydev_override import overrides
import weakref
from _pydev_bundle._pydev_completer import extract_token_and_qualifier
from _pydevd_bundle._debug_adapter.pydevd_schema import VariablesResponseBody, \
from _pydevd_bundle._debug_adapter import pydevd_base_schema, pydevd_schema
from _pydevd_bundle.pydevd_net_command import NetCommand
from _pydevd_bundle.pydevd_xml import ExceptionOnEvaluate
from _pydevd_bundle.pydevd_constants import ForkSafeLock, NULL
from _pydevd_bundle.pydevd_daemon_thread import PyDBDaemonThread
from _pydevd_bundle.pydevd_thread_lifecycle import pydevd_find_thread_by_id, resume_threads
from _pydevd_bundle.pydevd_dont_trace_files import PYDEV_FILE
import dis
import pydevd_file_utils
import itertools
from urllib.parse import quote_plus, unquote_plus
import pydevconsole
from _pydevd_bundle import pydevd_vars, pydevd_io, pydevd_reload
from _pydevd_bundle import pydevd_bytecode_utils
from _pydevd_bundle import pydevd_xml
from _pydevd_bundle import pydevd_vm_type
import sys
import traceback
from _pydevd_bundle.pydevd_utils import quote_smart as quote, compare_object_attrs_key, \
from _pydev_bundle import pydev_log, fsnotify
from _pydev_bundle.pydev_log import exception as pydev_log_exception
from _pydev_bundle import _pydev_completer
from pydevd_tracing import get_exception_traceback_str
from _pydevd_bundle import pydevd_console
from _pydev_bundle.pydev_monkey import disable_trace_thread_modules, enable_trace_thread_modules
from io import StringIO
from _pydevd_bundle.pydevd_comm_constants import *  # @UnusedWildImport
@silence_warnings_decorator
def internal_get_frame(dbg, seq, thread_id, frame_id):
    """ Converts request into python variable """
    try:
        frame = dbg.find_frame(thread_id, frame_id)
        if frame is not None:
            hidden_ns = pydevconsole.get_ipython_hidden_vars()
            xml = '<xml>'
            xml += pydevd_xml.frame_vars_to_xml(frame.f_locals, hidden_ns)
            del frame
            xml += '</xml>'
            cmd = dbg.cmd_factory.make_get_frame_message(seq, xml)
            dbg.writer.add_command(cmd)
        else:
            cmd = dbg.cmd_factory.make_error_message(seq, 'Frame not found: %s from thread: %s' % (frame_id, thread_id))
            dbg.writer.add_command(cmd)
    except:
        cmd = dbg.cmd_factory.make_error_message(seq, 'Error resolving frame: %s from thread: %s' % (frame_id, thread_id))
        dbg.writer.add_command(cmd)