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
def internal_set_expression_json(py_db, request, thread_id):
    arguments = request.arguments
    expression = arguments.expression
    frame_id = arguments.frameId
    value = arguments.value
    fmt = arguments.format
    if hasattr(fmt, 'to_dict'):
        fmt = fmt.to_dict()
    frame = py_db.find_frame(thread_id, frame_id)
    exec_code = '%s = (%s)' % (expression, value)
    result = pydevd_vars.evaluate_expression(py_db, frame, exec_code, is_exec=True)
    is_error = isinstance(result, ExceptionOnEvaluate)
    if is_error:
        _set_expression_response(py_db, request, result, error_message='Error executing: %s' % (exec_code,))
        return
    frame_tracker = py_db.suspended_frames_manager.get_frame_tracker(thread_id)
    if frame_tracker is None:
        _set_expression_response(py_db, request, result, error_message='Thread id: %s is not current thread id.' % (thread_id,))
        return
    result = pydevd_vars.evaluate_expression(py_db, frame, expression, is_exec=False)
    variable = frame_tracker.obtain_as_variable(expression, result, frame=frame)
    var_data = variable.get_var_data(fmt=fmt)
    body = pydevd_schema.SetExpressionResponseBody(value=var_data['value'], variablesReference=var_data.get('variablesReference', 0), type=var_data.get('type'), presentationHint=var_data.get('presentationHint'), namedVariables=var_data.get('namedVariables'), indexedVariables=var_data.get('indexedVariables'))
    variables_response = pydevd_base_schema.build_response(request, kwargs={'body': body})
    py_db.writer.add_command(NetCommand(CMD_RETURN, 0, variables_response, is_json=True))